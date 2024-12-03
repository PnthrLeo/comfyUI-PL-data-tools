import json
from pathlib import Path

import numpy as np
import torch

from .utils import (generate_clip_features_json, get_image_and_mask,
                    get_image_clip_embeddings)


class CloseImagesSearcher:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "clip_vision": ("CLIP_VISION",),
                "path_to_images_folder": ("STRING", {
                    "multiline": True,
                    "default": "/path/to/folder/with/images"
                }),
                "embeddings_database_name": ("STRING", {
                    "multiline": False,
                    "default": "database base"
                }),
            },
            "optional": {
                "path_to_masks_folder": ("STRING", {
                    "multiline": True,
                    "default": "path/to/folder/with/masks"
                }),
                "path_to_embeddings_databases": ("STRING", {
                    "multiline": True,
                    "default": "path/to/folder/with/embeddings/databases"
                }),
                "offset": ("INT", {
                    "default": 0,
                    "min": 0,
                    "step": 1,
                    "display": "number"
                }),
                "num_of_similar_images": ("INT", {
                    "default": 5,
                    "min": 0,
                    "step": 1,
                    "display": "number"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "INT")
    RETURN_NAMES = ("images", "masks", "scores")
    FUNCTION = "get_5_similar_images"
    CATEGORY = "PL Data Tools"

    def get_5_similar_images(self, image, clip_vision, path_to_images_folder,
                             embeddings_database_name, path_to_masks_folder,
                             path_to_embeddings_databases, offset,
                             num_of_similar_images):
        path_to_images_folder = Path(path_to_images_folder)
        path_to_masks_folder = Path(path_to_masks_folder)
        path_to_embeddings_databases = Path(path_to_embeddings_databases)

        if (not path_to_embeddings_databases.exists() or
                path_to_embeddings_databases == Path(".")):
            path_to_embeddings_databases = (Path(__file__).parent /
                                            'embeddings_databases')
            if not path_to_embeddings_databases.exists():
                path_to_embeddings_databases.mkdir()

        path_to_embeddings_db = (path_to_embeddings_databases /
                                 (embeddings_database_name + '.json'))

        if not path_to_embeddings_db.exists():
            generate_clip_features_json(clip_vision, path_to_images_folder,
                                        path_to_embeddings_db)

        image_embeds = get_image_clip_embeddings(clip_vision, image)

        with open(path_to_embeddings_db, 'r') as f:
            clip_features = json.load(f)

        distances = []
        for idx, (_, features) in enumerate(clip_features):
            distance = (np.dot(image_embeds, features) /
                        (np.linalg.norm(image_embeds) *
                         np.linalg.norm(features)))
            distances.append((idx, distance))

        distances = sorted(distances, key=lambda x: x[1], reverse=True)

        images = []
        masks = []
        scores = []

        for idx, distance in distances[offset:offset+num_of_similar_images]:
            file_name, _ = clip_features[idx]
            image, mask = get_image_and_mask(file_name, path_to_images_folder,
                                             path_to_masks_folder)
            images.append(image)
            masks.append(mask)
            scores.append(distance)

        images = np.concatenate(images, axis=0)
        print(images.shape)
        masks = np.concatenate(masks, axis=0)
        print(masks.shape)

        return torch.tensor(images), torch.tensor(masks), torch.tensor(scores)
