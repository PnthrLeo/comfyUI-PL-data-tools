from .areas_generator import AreasGenerator
from .batch_image_getter import BatchImageGetter
from .close_images_searcher import CloseImagesSearcher


NODE_CLASS_MAPPINGS = {
    "CloseImagesSearcher": CloseImagesSearcher,
    "AreasGenerator": AreasGenerator,
    "BatchImageGetter": BatchImageGetter
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "CloseImagesSearcher": "Close Images Searcher",
    "AreasGenerator": "Areas Generator",
    "BatchImageGetter": "Batch Image Getter"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
