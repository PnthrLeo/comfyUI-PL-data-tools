import torch
from typing import Tuple


class BatchImageGetter:
    """This node is used to get a single image from a tensor of images by
    index.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "idx": ("INT", {
                    "default": 0,
                    "min": 0,
                    "step": 1,
                    "display": "number"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image", )
    FUNCTION = "get_image_by_idx"
    CATEGORY = "PL Data Tools"

    def get_image_by_idx(self, images: torch.Tensor, idx: int
                         ) -> Tuple[torch.Tensor]:

        if idx >= len(images):
            raise ValueError(f'Index {idx} is out of bounds for the list of '
                             f'images. The list has {len(images)} elements. '
                             'Please provide an index between 0 '
                             f'and {len(images) - 1}.')

        return images[idx].unsqueeze(0),
