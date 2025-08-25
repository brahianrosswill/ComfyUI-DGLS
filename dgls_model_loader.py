import comfy.sd
import folder_paths
import torch
from safetensors.torch import load_file
import comfy.model_detection
import gc
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
# MODEL LOADER by obisin
# =============================================================================

def extract_layers_from_comfy_model(model_patcher):
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
                    print(f"Block {i} forward signature: {sig}")
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


def add_to_layers_method(model_patcher):
    """Add to_layers method to model so it works like your training code"""

    def to_layers():
        return extract_layers_from_comfy_model(model_patcher)

    # Add the method to the model_patcher so you can call model.to_layers()
    model_patcher.to_layers = to_layers
    return model_patcher


class DGLSModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        model_list = folder_paths.get_filename_list("unet") + folder_paths.get_filename_list("diffusion_models")#+ folder_paths.get_filename_list("checkpoints")
        return {
            "required": {
                "model_name": (model_list, {"default": model_list[0] if model_list else ""}),
                "model_type": (["default", "hunyuan", "unet"], {"default": "default"}),
                "cast_dtype": (["default", "fp32", "fp16", "bf16", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],
                                 {"default": "default"}),
                "clear_model_cache": ("BOOLEAN", {"default": False,
                                                  "tooltip": "Force reload model from disk, ignoring ComfyUI's model cache"}),
                "verbose": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "nuke_all_caches": ("BOOLEAN", {"default": False,
                                                "tooltip": "Clear all ComfyUI caches"}),
                "custom_ckpt_path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("MODEL", "LAYERS")
    RETURN_NAMES = ("model", "layers")
    FUNCTION = "load_dgls_model"
    CATEGORY = "loaders"
    TITLE = "DGLS Model Loader"

    def load_dgls_model(self, model_name, model_type, cast_dtype,  verbose, clear_model_cache=False, nuke_all_caches=False, custom_ckpt_path="", override_model_type="auto"):


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

        model_patcher = add_to_layers_method(model_patcher)

        # Extract layers
        layers = model_patcher.to_layers()

        if verbose:
            print(f" Model loaded successfully")
            print(f"Total layers extracted: {len(layers)}")

        return (model_patcher, layers)

    def clear_model_cache_only(self):
        """Safely clear only model-related caches, preserve node execution cache"""
        import comfy.model_management as model_management

        if hasattr(self, 'verbose') and self.verbose:
            print("🔄 Clearing model cache only (safe mode)...")

        # Only clear models, not execution cache
        model_management.unload_all_models()
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

        # Clear execution cache (dangerous part)
        try:
            import execution
            # This is the risky part - affects ALL nodes
            if hasattr(execution, 'PromptServer'):
                server = execution.PromptServer.instance
                if hasattr(server, 'cache'):
                    server.cache.outputs.clear()
                    server.cache.ui.clear()
                    server.cache.objects.clear()
                    print("  Execution cache cleared - other nodes may reload")
        except Exception as e:
            print(f"  Could not clear execution cache: {e}")
