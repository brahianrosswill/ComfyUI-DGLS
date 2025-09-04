import comfy.sd
import folder_paths
import torch
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

COMPATIBILITY: Diffusion models and UNets only. Not compatible with full checkpoints yett.

Author: obisin
"""

# =============================================================================
# EXTRACTION by obisin
# =============================================================================

def extract_layers_from_comfy_model(model_patcher, verbose):
    """Extract ComfyUI's layers from different model architectures"""

    if hasattr(model_patcher.model, 'diffusion_model'):
        dm = model_patcher.model.diffusion_model

        layers = []

        # Method 1: Standard blocks (WAN, SDXL, etc.)
        if hasattr(dm, 'blocks') and len(dm.blocks) > 0:
            for i, block in enumerate(dm.blocks):
                layers.append(block)
            print(f"Extracted {len(layers)} blocks from diffusion model")
            return layers

        # Method 2: Flux DiT layers (double_blocks + single_blocks)
        if hasattr(dm, 'double_blocks') or hasattr(dm, 'single_blocks'):
            if hasattr(dm, 'double_blocks'):
                for i, block in enumerate(dm.double_blocks):
                    layers.append(block)
                    # Debug: Check the forward signature
                    import inspect
                    sig = inspect.signature(block.forward)
                print(f"Added {len(dm.double_blocks)} double_blocks")

            if hasattr(dm, 'single_blocks'):
                for i, block in enumerate(dm.single_blocks):
                    layers.append(block)
                print(f"Added {len(dm.single_blocks)} single_blocks")

            print(f"Extracted {len(layers)} Flux DiT layers")
            return layers

        # Method 3: SD3 DiT layers
        if hasattr(dm, 'joint_blocks'):
            for i, block in enumerate(dm.joint_blocks):
                layers.append(block)
            print(f"Extracted {len(layers)} SD3 joint_blocks")
            return layers

        # Method 4: HunyuanVideo layers
        if hasattr(dm, 'blocks') and hasattr(dm, 'single_blocks'):
            # HunyuanVideo has both blocks and single_blocks
            for i, block in enumerate(dm.blocks):
                layers.append(block)
            for i, block in enumerate(dm.single_blocks):
                layers.append(block)
            print(f"Extracted {len(layers)} HunyuanVideo layers (blocks + single_blocks)")
            return layers

        # Method 5: Transformer layers (generic DiT models)
        if hasattr(dm, 'transformer_blocks'):
            for i, block in enumerate(dm.transformer_blocks):
                layers.append(block)
            print(f"Extracted {len(layers)} transformer_blocks")
            return layers

        # Method 6: Generic layer extraction by common names
        layer_names = ['layers', 'encoders', 'decoders', 'blocks', 'modules']
        for attr_name in layer_names:
            if hasattr(dm, attr_name):
                attr = getattr(dm, attr_name)
                if hasattr(attr, '__iter__') and not isinstance(attr, str):
                    try:
                        for i, block in enumerate(attr):
                            if isinstance(block, torch.nn.Module):
                                layers.append(block)
                        if layers:
                            print(f"Extracted {len(layers)} layers from {attr_name}")
                            return layers
                    except:
                        continue

        # Method 7: Search for block-like modules
        for name, module in dm.named_children():
            if isinstance(module, torch.nn.Module):
                # Look for modules that are likely to be computation blocks
                if any(keyword in name.lower() for keyword in ['block', 'layer', 'transformer', 'attention', 'mlp']):
                    # Check if it's a container of blocks or a single block
                    if hasattr(module, '__iter__') and not isinstance(module, torch.nn.Parameter):
                        try:
                            for sub_module in module:
                                if isinstance(sub_module, torch.nn.Module):
                                    layers.append(sub_module)
                        except:
                            layers.append(module)
                    else:
                        layers.append(module)

        if layers:
            print(f"Extracted {len(layers)} layers using pattern matching")
            return layers

        # Method 8: Last resort - extract all meaningful modules
        print(" Using fallback: extracting all major modules")
        skip_modules = ['input_layer', 'output_layer', 'norm', 'embedding', 'pos_embed', 'patch_embed']
        for name, module in dm.named_children():
            if isinstance(module, torch.nn.Module) and name not in skip_modules:
                layers.append(module)
                print(f"  Added module: {name} ({type(module).__name__})")

        if layers:
            print(f"Extracted {len(layers)} modules using fallback method")
            return layers
        else:
            print(" No layers found in diffusion model")
            # Debug: show what's available
            print("Available attributes:")
            for name in dir(dm):
                if not name.startswith('_') and hasattr(dm, name):
                    attr = getattr(dm, name)
                    if isinstance(attr, (torch.nn.Module, torch.nn.ModuleList, list)):
                        print(f"  {name}: {type(attr)}")
            return []
    else:
        print(" No diffusion_model found")
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
                "cast_dtype": (["default", "fp32", "fp16", "bf16", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],{"default": "default"}),
                "clear_model_cache": ("BOOLEAN", {"default": False, "tooltip": "Force reload model from disk, ignoring ComfyUI's model cache"}),
                "verbose": ("BOOLEAN", {"default": False}),
            },
            "optional": {"nuke_all_caches": ("BOOLEAN", {"default": False, "tooltip": "CAUTION: Aggressive Clear all ComfyUI caches"}),
                # "custom_ckpt_path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("MODEL", "LAYERS")
    RETURN_NAMES = ("model", "layers")
    FUNCTION = "load_dgls_model"
    CATEGORY = "loaders"
    TITLE = "DGLS Model Loader"

    def load_dgls_model(self, model_name, model_type, cast_dtype,  verbose, clear_model_cache, nuke_all_caches=False, custom_ckpt_path="", override_model_type="auto"):

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
            print(f"Weight dtype: {cast_dtype}")
            print(f"Override model type: {override_model_type}")


        # Set up model options for dtype control
        model_options = {}
        if cast_dtype == "fp32":
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
                            param.data.clone())
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
