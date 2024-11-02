from .areas_generator import AreasGenerator
from .close_images_searcher import CloseImagesSearcher

NODE_CLASS_MAPPINGS = {
    "CloseImagesSearcher": CloseImagesSearcher,
    "AreasGenerator": AreasGenerator
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "CloseImagesSearcher": "Close Images Searcher",
    "AreasGenerator": "Areas Generator"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
