from .dynamic_swapping_loader import DynamicSwappingLoader
from .dgls_model_loader import DGLSModelLoader
from. dgls_cleanup import DGLSCleanup

NODE_CLASS_MAPPINGS = {
    "DGLSModelLoader": DGLSModelLoader,
    "DynamicSwappingLoader": DynamicSwappingLoader,
    "DGLSCleanup": DGLSCleanup
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DGLSModelLoader": "DGLS Model Loader",
    "DynamicSwappingLoader": "DGLS Swapping Loader",
    "DGLSCleanup": "DGLS Cleanup"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
