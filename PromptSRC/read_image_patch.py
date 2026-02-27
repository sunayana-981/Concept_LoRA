# Add to the top of tools.py
import os
from PIL import Image

# Global registry for in-memory images
_IMAGE_REGISTRY = {}

def register_image(path, pil_image):
    """Register a PIL image for a virtual path"""
    _IMAGE_REGISTRY[path] = pil_image

def read_image(path):
    """Read image from path using ``PIL.Image``.
    
    Args:
        path (str): path to an image or virtual path.
        
    Returns:
        PIL image
    """
    # Check if this is a registered in-memory image
    if path in _IMAGE_REGISTRY:
        return _IMAGE_REGISTRY[path].convert("RGB")
    
    # Otherwise, read from disk as usual
    return Image.open(path).convert("RGB")
