import comfy.sd
import folder_paths
import torch
import torch.nn as nn
from safetensors.torch import load_file
import comfy.model_detection
import gc
import comfy.utils
import comfy.model_management
import comfy.sd
import folder_paths
import torch
import comfy.model_management
import comfy.utils
import comfy.lora
import comfy.float
import collections
from comfy.model_patcher import get_key_weight, string_to_seed
"""
DGLS Model Loader for ComfyUI by obisin
============================
Universal diffusion model loader with layer extraction for memory optimization.

Extracts computation layers from UNet/DiT architectures for dynamic GPU swapping.
Supports Cosmos, Flux, Wan2.1, Wan2.2, HunyuanVideo, and generic transformer blocks.

COMPATIBILITY: Diffusion models and UNets only. Not compatible with full checkpoints yet.

Author: obisin
"""

# =============================================================================
# FULL CAST SYSTEM by obisin
# =============================================================================

def _owner_and_key(root_module, dotted_name):
    """Helper to find the owner module and key for a parameter/buffer"""
    if "." not in dotted_name:
        return root_module, dotted_name
    owner_path, key = dotted_name.rsplit(".", 1)
    owner = root_module
    for part in owner_path.split("."):
        owner = getattr(owner, part)
    return owner, key

class FullCastHandler:
    def __init__(self):
        self.dtype_map = {
            'fp32': torch.float32,
            'fp16': torch.float16,
            'bf16': torch.bfloat16,
            'fp8_e4m3': torch.float8_e4m3fn if hasattr(torch, 'float8_e4m3fn') else None,
            'fp8_e5m2': torch.float8_e5m2 if hasattr(torch, 'float8_e5m2') else None,
            'nf4': 'nf4',  # Special handling
            'fp4': 'fp4'   # Special handling
        }

    def cast_model_to_dtype(self, model_patcher, target_dtype_str, verbose=False):
        """Cast entire model to target dtype"""
        if target_dtype_str == "disabled":
            return model_patcher

        target_dtype = self.dtype_map.get(target_dtype_str)
        if target_dtype is None:
            if verbose:
                print(f"Unsupported target dtype: {target_dtype_str}")
            return model_patcher

        if verbose:
            print(f"Full casting model to {target_dtype_str}...")

        model = model_patcher.model

        # Handle 4-bit quantization
        if target_dtype_str in ['nf4', 'fp4']:
            return self._cast_to_4bit(model_patcher, target_dtype_str, verbose)

        # Regular dtype casting
        casted_params = 0
        casted_buffers = 0

        with torch.inference_mode(False), torch.no_grad():
            # Cast parameters in-place
            for name, param in model.named_parameters(recurse=True):
                if param is None or not torch.is_floating_point(param):
                    continue

                if param.dtype != target_dtype:
                    param.data = param.data.to(dtype=target_dtype, device=param.device)
                    casted_params += 1

            # Cast buffers in-place
            for name, buffer in model.named_buffers(recurse=True):
                if buffer is None or not torch.is_floating_point(buffer):
                    continue

                # Keep batchnorm running stats in fp32 for stability
                if name.endswith("running_mean") or name.endswith("running_var"):
                    continue

                if buffer.dtype != target_dtype:
                    buffer.data = buffer.data.to(dtype=target_dtype, device=buffer.device)
                    casted_buffers += 1

        if verbose:
            print(f"✓ Full cast complete: {casted_params} parameters, {casted_buffers} buffers cast to {target_dtype_str}")

        return model_patcher

    def _cast_to_4bit(self, model_patcher, quant_type, verbose=False):
        """Cast model to 4-bit quantization using bitsandbytes"""
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise RuntimeError("bitsandbytes required for 4-bit quantization. Install with: pip install bitsandbytes")

        if verbose:
            print(f"Applying 4-bit quantization ({quant_type})...")

        model = model_patcher.model
        quantized_params = 0

        with torch.inference_mode(False), torch.no_grad():
            for name, param in model.named_parameters(recurse=True):
                if param is None or not torch.is_floating_point(param):
                    continue

                try:
                    # Quantize and immediately dequantize in-place
                    quantized, quant_state = bnb.functional.quantize_4bit(
                        param.data, blocksize=64, quant_type=quant_type
                    )
                    param.data = bnb.functional.dequantize_4bit(quantized, quant_state)
                    quantized_params += 1

                except Exception as e:
                    if verbose:
                        print(f"Failed to quantize {name}: {e}")
                    continue

        if verbose:
            print(f"✓ 4-bit quantization complete: {quantized_params} parameters quantized with {quant_type}")

        return model_patcher

# =============================================================================
# EXTRACTION by obisin
# =============================================================================

def extract_layers_from_comfy_model(model_patcher, verbose):
    """Vectorized layer extraction with optimized single-pass detection"""

    if not hasattr(model_patcher.model, 'diffusion_model'):
        if verbose:
            print(" No diffusion_model found")
        return []

    dm = model_patcher.model.diffusion_model

    # Priority-ordered layer attributes (most common architectures first)
    primary_attrs = [
        ('blocks', 'Standard blocks (WAN, SDXL, etc.)'),
        ('double_blocks', 'Flux DiT double_blocks'),
        ('single_blocks', 'Flux DiT single_blocks'), 
        ('joint_blocks', 'SD3 DiT joint_blocks'),
        ('transformer_blocks', 'Generic DiT transformer_blocks')
    ]

    # Single pass extraction for primary attributes
    found_layers = []
    found_sources = []

    for attr_name, description in primary_attrs:
        if hasattr(dm, attr_name):
            attr = getattr(dm, attr_name)
            if attr and len(attr) > 0:
                # Direct list conversion instead of enumeration loop
                layer_list = list(attr)
                if layer_list:
                    found_layers.extend(layer_list)
                    found_sources.append(f"{len(layer_list)} {attr_name}")

    # Special case: HunyuanVideo (both blocks and single_blocks)
    if (hasattr(dm, 'blocks') and hasattr(dm, 'single_blocks') and 
        not found_layers):  # Only if nothing found yet
        blocks = list(dm.blocks) if dm.blocks else []
        single_blocks = list(dm.single_blocks) if dm.single_blocks else []
        if blocks or single_blocks:
            found_layers.extend(blocks + single_blocks)
            found_sources.append(f"HunyuanVideo: {len(blocks)} blocks + {len(single_blocks)} single_blocks")

    if found_layers:
        if verbose:
            print(f"Extracted {len(found_layers)} layers from: {', '.join(found_sources)}")
        return found_layers

    # Fast fallback using dir() filtering instead of individual hasattr calls
    fallback_attrs = [name for name in dir(dm) 
                     if not name.startswith('_') and 
                     any(keyword in name.lower() for keyword in 
                         ['layer', 'block', 'encoder', 'decoder', 'module'])]

    for attr_name in fallback_attrs:
        try:
            attr = getattr(dm, attr_name, None)
            if hasattr(attr, '__iter__') and not isinstance(attr, (str, torch.nn.Parameter)):
                # Vectorized module filtering
                modules = [item for item in attr if isinstance(item, torch.nn.Module)]
                if modules:
                    if verbose:
                        print(f"Extracted {len(modules)} layers from {attr_name}")
                    return modules
        except:
            continue

    # Final fallback: named_children with pattern matching
    skip_modules = {'input_layer', 'output_layer', 'norm', 'embedding', 'pos_embed', 'patch_embed'}
    pattern_keywords = {'block', 'layer', 'transformer', 'attention', 'mlp'}
    
    fallback_layers = []
    for name, module in dm.named_children():
        if (isinstance(module, torch.nn.Module) and 
            name not in skip_modules and
            any(keyword in name.lower() for keyword in pattern_keywords)):
            
            # Check if it's a container of blocks
            if hasattr(module, '__iter__') and not isinstance(module, torch.nn.Parameter):
                try:
                    sub_modules = [sub for sub in module if isinstance(sub, torch.nn.Module)]
                    fallback_layers.extend(sub_modules)
                except:
                    fallback_layers.append(module)
            else:
                fallback_layers.append(module)

    if fallback_layers:
        if verbose:
            print(f"Extracted {len(fallback_layers)} layers using pattern matching")
        return fallback_layers

    # Last resort: all meaningful modules
    if verbose:
        print(" Using final fallback: extracting all major modules")
    
    final_layers = []
    for name, module in dm.named_children():
        if isinstance(module, torch.nn.Module) and name not in skip_modules:
            final_layers.append(module)
            if verbose:
                print(f"  Added module: {name} ({type(module).__name__})")

    if final_layers:
        if verbose:
            print(f"Extracted {len(final_layers)} modules using final fallback")
        return final_layers

    # Debug information only if verbose
    if verbose:
        print(" No layers found in diffusion model")
        print("Available attributes:")
        for name in dir(dm):
            if not name.startswith('_'):
                attr = getattr(dm, name, None)
                if isinstance(attr, (torch.nn.Module, torch.nn.ModuleList, list)):
                    print(f"  {name}: {type(attr)}")
    
    return []


def add_to_layers_method(model_patcher,verbose):
    """Add to_layers method to model so it works like your training code"""

    def to_layers():
        return extract_layers_from_comfy_model(model_patcher, verbose=verbose)

    # Add the method to the model_patcher so you can call model.to_layers()
    model_patcher.to_layers = to_layers
    return model_patcher


# =============================================================================
# MODEL LOADER by obisin
# =============================================================================


class DGLSModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        model_list = folder_paths.get_filename_list("unet") + folder_paths.get_filename_list("diffusion_models")#+ folder_paths.get_filename_list("checkpoints")
        return {
            "required": {
                "model_name": (model_list, {"default": model_list[0] if model_list else ""}),
                "model_type": (["default", "hunyuan", "unet"], {"default": "default", "tooltip": "Unet is just for Debugging, its a legacy loader- Use default"}),
                "cast_dtype": (["default", "disabled", "fp32", "fp16", "bf16", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],{"default": "default", "tooltip": "ComfyUI casting system. Use 'disabled' to turn off ComfyUI casting and use full_cast instead"}),
                "full_cast": (["disabled", "fp32", "fp16", "bf16", "fp8_e4m3", "fp8_e5m2", "nf4", "fp4"], {"default": "disabled", "tooltip": "Full cast system - cast entire model to target dtype including fp4. Works independently of ComfyUI's system. Only cast in f8 or f4 if you have the kernals for it."}),
                "clear_model_cache": ("BOOLEAN", {"default": False, "tooltip": "Force reload model from disk, ignoring ComfyUI's model cache"}),
                "verbose": ("BOOLEAN", {"default": False, "tooltip": "This will slow down inference"}),
            },
            "optional": {
                "nuke_all_caches": ("BOOLEAN", {"default": False, "tooltip": "CAUTION: Aggressive Clear all ComfyUI caches"}),
                # "custom_ckpt_path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("MODEL", "LAYERS")
    RETURN_NAMES = ("model", "layers")
    FUNCTION = "load_dgls_model"
    CATEGORY = "loaders"
    TITLE = "DGLS Model Loader"

    def load_dgls_model(self, model_name, model_type, cast_dtype, full_cast, verbose, clear_model_cache, nuke_all_caches=False, custom_ckpt_path="", override_model_type="auto"):

        self.verbose = verbose
        # Cache clearing logic
        if nuke_all_caches:
            self.nuke_all_caches()
        elif clear_model_cache:
            self.clear_model_cache_only()

        if custom_ckpt_path.strip():
            model_path = custom_ckpt_path.strip()
        else:
            try:
                model_path = folder_paths.get_full_path('unet', model_name)
            except:
                model_path = folder_paths.get_full_path('diffusion_models', model_name)

        if verbose:
            print(f"Loading model using DGLS loader...")
            print(f"Model path: {model_path}")
            print(f"ComfyUI cast dtype: {cast_dtype}")
            print(f"Full cast dtype: {full_cast}")
            print(f"Override model type: {override_model_type}")


        # Set up model options for dtype control
        model_options = {}
        if cast_dtype == "disabled":
            # Disable ComfyUI's casting system completely
            model_options = {}
        elif cast_dtype == "fp32":
            model_options["dtype"] = torch.float32
        elif cast_dtype == "fp16":
            model_options["dtype"] = torch.float16
        elif cast_dtype == "bf16":
            model_options["dtype"] = torch.bfloat16
        elif cast_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif cast_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif cast_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2
        elif cast_dtype == "default":
            model_options = {}

        if verbose and model_options:
            print(f"Model options: {model_options}")

        if verbose:
            if model_path.endswith('.safetensors'):
                from safetensors.torch import load_file
                sd = load_file(model_path)
            else:
                sd = torch.load(model_path, map_location='cpu')

            print(f"Sample state dict keys: {list(sd.keys())[:5]}")
            detected_config = comfy.model_detection.model_config_from_unet(sd, "")
            detected_type = type(detected_config).__name__
            print(f"ComfyUI auto-detected: {detected_type}")

        if model_type == "hunyuan":
            # Use state dict method for HunyuanVideo (avoids disconnection)
            if model_path.endswith('.safetensors'):
                from safetensors.torch import load_file
                sd = load_file(model_path)
            else:
                sd = torch.load(model_path, map_location='cpu')

            model_patcher = comfy.sd.load_diffusion_model_state_dict(sd, model_options=model_options)

        elif model_type == "unet":
            # import comfy.model_management
            unet_path = model_path
            model_patcher = comfy.sd.load_unet(unet_path)
        else:
            model_patcher = comfy.sd.load_diffusion_model(model_path, model_options=model_options)

        model_patcher = self.ensure_model_ready(model_patcher, verbose=verbose)
        # fixed = self.normalize_inference_params(model_patcher, verbose=verbose)
        # model_patcher = self.normalize_inference_model_patcher(model_patcher)

        # Apply full cast system if enabled
        if full_cast != "disabled":
            if verbose:
                print(f"Applying full cast to {full_cast}...")
            full_cast_handler = FullCastHandler()
            model_patcher = full_cast_handler.cast_model_to_dtype(model_patcher, full_cast, verbose=verbose)

        model_patcher = add_to_layers_method(model_patcher, verbose=verbose)
        layers = model_patcher.to_layers()
        if verbose:
            print(f" Model loaded successfully")
            print(f"Total layers extracted: {len(layers)}")

        return (model_patcher, layers)

    # =============================================================================
    # TENSOR FIXES by obisin
    # =============================================================================

    def ensure_model_ready(self, model_patcher, verbose=False):
        """Ensure model ready without large allocation"""
        if not hasattr(model_patcher.model, 'current_weight_patches_uuid'):
            model_patcher.model.current_weight_patches_uuid = model_patcher.patches_uuid

        # Fix version counters
        with torch.inference_mode(False), torch.no_grad():
            for name, param in model_patcher.model.named_parameters():
                if param is not None:
                    try:
                        _ = param._version
                    except:
                        comfy.utils.set_attr_param(
                            model_patcher.model,
                            name,
                            param.data)#.clone()
        return model_patcher

    def normalize_inference_model_patcher(self, model_patcher, verbose=False, attr_names=None):

        # Step 2: Apply object patches (from patch_model)
        for k in model_patcher.object_patches:
            old = comfy.utils.set_attr(model_patcher.model, k, model_patcher.object_patches[k])
            if k not in model_patcher.object_patches_backup:
                model_patcher.object_patches_backup[k] = old

        # Step 3: Apply weight patches WITHOUT the splitting logic
        # This replicates the patching part of load() but stops before module iteration
        if len(model_patcher.patches) > 0:
            print(f"Applying {len(model_patcher.patches)} weight patches...")

            # Unpatch any existing hooks first (as load() does)
            model_patcher.unpatch_hooks()

            # Apply each weight patch
            for key in model_patcher.patches:
                # Determine device for this weight based on ComfyUI's logic
                device_to = model_patcher.load_device

                # Get the weight
                weight, set_func, convert_func = get_key_weight(model_patcher.model, key)

                # Backup original weight
                if key not in model_patcher.backup:
                    model_patcher.backup[key] = collections.namedtuple('Dimension', ['weight', 'inplace_update'])(
                        weight.to(device=model_patcher.offload_device, copy=False),
                        False
                    )

                # Move to appropriate device and convert to float32 for patching
                if device_to is not None:
                    temp_weight = comfy.model_management.cast_to_device(weight, device_to, torch.float32, copy=True)
                else:
                    temp_weight = weight.to(torch.float32, copy=True)

                if convert_func is not None:
                    temp_weight = convert_func(temp_weight, inplace=True)

                # Apply patches
                out_weight = comfy.lora.calculate_weight(model_patcher.patches[key], temp_weight, key)

                # Apply the patched weight
                # if set_func is None:
                #     out_weight = comfy.float.stochastic_rounding(out_weight, weight.dtype, seed=string_to_seed(key))
                #     comfy.utils.set_attr_param(model_patcher.model, key, out_weight)
                # else:
                #     set_func(out_weight, inplace_update=False, seed=string_to_seed(key))

                # Apply the patched weight with version counter preservation
                if set_func is None:
                    # Use your working healing approach instead of stochastic_rounding
                    out_weight = comfy.model_management.cast_to_device(out_weight, out_weight.device, weight.dtype,
                                                                       copy=True)
                    comfy.utils.set_attr_param(model_patcher.model, key, out_weight)
                else:
                    # For set_func, ensure the tensor has version tracking
                    with torch.inference_mode(False):
                        healed_weight = comfy.model_management.cast_to_device(out_weight, out_weight.device,
                                                                              weight.dtype, copy=True)
                        set_func(healed_weight, inplace_update=False, seed=string_to_seed(key))

            # Step 4: Apply model patches to device (from load())
            model_patcher.model_patches_to(model_patcher.load_device)
            model_patcher.model_patches_to(model_patcher.model.model_dtype())

            # Step 5: Set model state (but don't do the module splitting)
            model_patcher.model.model_lowvram = False
            model_patcher.model.lowvram_patch_counter = 0
            model_patcher.model.device = model_patcher.load_device
            model_patcher.model.current_weight_patches_uuid = model_patcher.patches_uuid

            # Step 6: Apply forced hooks if any
            if hasattr(model_patcher, 'forced_hooks') and model_patcher.forced_hooks:
                model_patcher.apply_hooks(model_patcher.forced_hooks, force_apply=True)

            # Step 7: Inject model
            model_patcher.inject_model()

        return model_patcher

    def normalize_inference_params(self, model_patcher, verbose=False, attr_names=None):
        """
        Ensure all params, buffers, and common forward-only tensor attrs are
        'normal' tensors (have a version counter) like Comfy does before patching.
        """
        model = model_patcher.model
        fixed_params = fixed_bufs = fixed_attrs = 0

        # helper: does this tensor have a tracked version counter?
        def tracks_version(t: torch.Tensor) -> bool:
            try:
                _ = t._version
                return True
            except Exception:
                return False

        # helper: force fresh storage
        def heal(t: torch.Tensor) -> torch.Tensor:
            return comfy.model_management.cast_to_device(t, t.device, t.dtype, copy=True)

        # common attr names seen in WAN/Flux-style blocks
        TENSOR_ATTRS = set(attr_names or [
            "modulation", "freqs", "pe", "vec", "norm", "scale",
            "pos_embed", "time_embed", "label_embed", "positional_encoding",
        ])

        with torch.inference_mode(False), torch.no_grad():
            # 1) Parameters: replace via Comfy utils so dotted names are handled
            for name, p in model.named_parameters(recurse=True):
                if p is None or tracks_version(p):
                    continue
                comfy.utils.set_attr_param(model, name, heal(p))
                fixed_params += 1

            # 2) re-register at the owner module, preserving persistence
            for name, b in model.named_buffers(recurse=True):
                if b is None or tracks_version(b):
                    continue
                owner, attr = name.rsplit(".", 1) if "." in name else ("", name)
                mod = comfy.utils.get_attr(model, owner) if owner else model
                # preserve persistent flag if the module had it non-persistent
                persistent = True
                nps = getattr(mod, "_non_persistent_buffers_set", None)
                if isinstance(nps, set) and attr in nps:
                    persistent = False
                mod.register_buffer(attr, heal(b), persistent=persistent)
                fixed_bufs += 1

            # 3) Plain tensor attributes used in forward that aren't params/buffers
            for m in model.modules():
                for an in TENSOR_ATTRS:
                    if not hasattr(m, an):
                        continue
                    t = getattr(m, an)
                    if isinstance(t, torch.Tensor) and not tracks_version(t):
                        setattr(m, an, heal(t))
                        fixed_attrs += 1

        if verbose:
            print(f"normalize_inference_params: params={fixed_params}, buffers={fixed_bufs}, attrs={fixed_attrs}")
        return fixed_params + fixed_bufs + fixed_attrs


    # =============================================================================
    # CACHE CLEARING by obisin
    # =============================================================================

    def clear_model_cache_only(self):
        """Safely clear only model-related caches, preserve node execution cache"""
        import comfy.model_management as model_management

        if hasattr(self, 'verbose') and self.verbose:
            print("Clearing model cache only...")

        model_management.unload_all_models()
        comfy.model_management.cleanup_models_gc()
        model_management.soft_empty_cache(True)

        # GPU cleanup
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()
        
        # CUDA IPC cleanup - complements GPU cleanup
        try:
            torch.cuda.ipc_collect()  # Clear CUDA IPC memory handles
        except Exception:
            pass  # Graceful fallback if IPC collect fails

        if hasattr(self, 'verbose') and self.verbose:
            print("✓ Model cache cleared safely")

    def nuke_all_caches(self):
        """Clear ALL caches - may break downstream nodes"""
        if hasattr(self, 'verbose') and self.verbose:
            print("Clearing ALL ComfyUI caches...")

        # Clear models first
        self.clear_model_cache_only()

        # Clear execution cache
        try:
            import execution
            # This is the risky part - affects ALL nodes
            if hasattr(execution, 'PromptServer'):
                server = execution.PromptServer.instance
                if hasattr(server, 'cache'):
                    server.cache.outputs.clear()
                    server.cache.ui.clear()
                    server.cache.objects.clear()
                    torch.cuda.synchronize()
                    print("  Execution cache cleared - other nodes may reload")
        except Exception as e:
            print(f"  Could not clear execution cache: {e}")
