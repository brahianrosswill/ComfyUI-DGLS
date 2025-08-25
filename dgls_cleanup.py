import torch
import gc
import threading

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


any_type = AnyType("*")


class DGLSCleanup:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trigger": (any_type, {}),
            },
            "optional": {
                "use_comfy_clear": ("BOOLEAN",
                                    {"default": True, "tooltip": "Use ComfyUI's built-in Clear Cache All function"}),
                "verbose": ("BOOLEAN", {"default": True}),
            },
            "hidden": {"unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO"}
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("output",)
    OUTPUT_NODE = True
    FUNCTION = "cleanup"
    CATEGORY = "loaders"
    TITLE = "DGLS Cleanup"

    def diagnose_persistent_state(self, verbose):
        """Check what's still in memory after cleanup"""
        if not verbose:
            return

        print("\n=== POST-CLEANUP DIAGNOSIS ===")

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1e9
            reserved = torch.cuda.memory_reserved(0) / 1e9
            print(f"CUDA memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

            if allocated > 0.5:
                print(f"WARNING: {allocated:.2f}GB still allocated after cleanup!")

        # Check ComfyUI's loaded models
        import comfy.model_management as model_management
        loaded = model_management.current_loaded_models
        print(f"ComfyUI loaded models: {len(loaded)}")

        print("==============================\n")

    def cleanup(self, trigger, use_comfy_clear=True, verbose=True, unique_id=None, extra_pnginfo=None):
        if verbose:
            print("DGLS Cache cleanup starting...")

        if use_comfy_clear:
            # Use ComfyUI's built-in cache clearing
            import comfy.model_management as model_management
            model_management.unload_all_models()
            model_management.soft_empty_cache(True)

            if verbose:
                print("ComfyUI cache cleared")

        # Additional CUDA cleanup
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()

        try:
            torch.cuda.ipc_collect()
        except:
            pass

        if verbose:
            print("DGLS Cache cleared")
            self.diagnose_persistent_state(verbose)

        return (trigger,)
