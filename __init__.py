from .dynamic_swapping_loader import DynamicSwappingLoader
from .dgls_model_loader import DGLSModelLoader

NODE_CLASS_MAPPINGS = {
    "DGLSModelLoader": DGLSModelLoader,
    "DynamicSwappingLoader": DynamicSwappingLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DGLSModelLoader": "DGLS Model Loader",
    "DynamicSwappingLoader": "DGLS Swapping Loader"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']