import cv2
import numpy as np
import torch


class AreasGenerator:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_width": ("INT", {
                    "default": 1024,
                    "min": 0,
                    "step": 1,
                    "display": "number"
                }),
                "image_height": ("INT", {
                    "default": 1024,
                    "min": 0,
                    "step": 1,
                    "display": "number"
                }),
                "basic_shape": (
                    ["circle", "rectangle"],
                ),
                "min_zone_width": ("INT", {
                    "default": 32,
                    "min": 0,
                    "step": 1,
                    "display": "number"
                }),
                "max_zone_width": ("INT", {
                    "default": 32,
                    "min": 0,
                    "step": 1,
                    "display": "number"
                }),
                "min_zone_height": ("INT", {
                    "default": 32,
                    "min": 0,
                    "step": 1,
                    "display": "number"
                }),
                "max_zone_height": ("INT", {
                    "default": 32,
                    "min": 0,
                    "step": 1,
                    "display": "number"
                }),
                "num_of_zones": ("INT", {
                    "default": 10,
                    "min": 0,
                    "step": 1,
                    "display": "number"
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "step": 1,
                    "display": "number"
                })
            }
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("Areas Mask", )
    FUNCTION = "generate_areas"
    CATEGORY = "PL Data Tools"

    def generate_areas(self, image_width, image_height, basic_shape,
                       min_zone_width, max_zone_width, min_zone_height,
                       max_zone_height, num_of_zones, seed):
        np.random.seed(seed)

        areas_mask = np.zeros((image_height, image_width))
        for _ in range(num_of_zones):
            if basic_shape == "circle":
                shape = self.generate_circle_image(
                    np.random.randint(min_zone_width, max_zone_width),
                    np.random.randint(min_zone_height, max_zone_height)
                )
            elif basic_shape == "rectangle":
                shape = self.generate_rectangle_image(
                    np.random.randint(min_zone_width, max_zone_width),
                    np.random.randint(min_zone_height, max_zone_height)
                )

            x = np.random.randint(0, image_width - shape.shape[1])
            y = np.random.randint(0, image_height - shape.shape[0])

            areas_mask = self.place_shape_on_image(areas_mask, shape, x, y)

        return torch.tensor(areas_mask)[None, None, ...]

    def generate_circle_image(self, image_width, image_height):
        image = np.zeros((image_height, image_width))
        half_width, half_height = image_width // 2, image_height // 2
        center = (half_width, half_height)
        radius = min(half_width, half_height)
        image = cv2.circle(image, center, radius, 1, -1)
        return image

    def generate_rectangle_image(self, image_width, image_height):
        image = np.ones((image_height, image_width))
        return image

    def place_shape_on_image(self, image, shape, x, y):
        image[y:y+shape.shape[0], x:x+shape.shape[1]] = \
            np.where(shape == 1, 1, image[y:y+shape.shape[0],
                                          x:x+shape.shape[1]])
        return image
