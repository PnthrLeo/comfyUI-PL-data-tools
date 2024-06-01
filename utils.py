import json
from pathlib import Path

import comfy.model_management as model_management
import torch
from comfy.clip_vision import Output, clip_preprocess
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm


def generate_clip_features_json(clip_vision, path_to_images_folder: Path,
                                output_json_path: Path):
    clip_features = []

    image_path_list = list(path_to_images_folder.glob('**/*.*'))

    model_management.load_model_gpu(clip_vision.patcher)
    print("--------- Generating embeddings for images ---------")
    for image_path in tqdm(image_path_list):
        image = Image.open(image_path)
        image_embeds = get_image_clip_embeddings(clip_vision, image)

        clip_features.append((str(image_path.name), image_embeds))

    with open(output_json_path, 'w') as f:
        json.dump(clip_features, f)


def get_image_clip_embeddings(clip_vision, image: Image):
    model_management.load_model_gpu(clip_vision.patcher)

    if type(image) is not torch.Tensor:
        image = image_to_tensor(image)
    image = image.to(clip_vision.load_device)

    pixel_values = clip_preprocess(image)

    out = clip_vision.model(pixel_values=pixel_values, intermediate_output=-2)

    outputs = Output()
    outputs["last_hidden_state"] = out[0].to(
        model_management.intermediate_device())
    outputs["image_embeds"] = out[2].to(
        model_management.intermediate_device())
    outputs["penultimate_hidden_states"] = out[1].to(
        model_management.intermediate_device())

    return outputs["image_embeds"].numpy().flatten().tolist()


def get_image_and_mask(file_name: str, path_to_images_folder: Path,
                       path_to_masks_folder: Path):
    image = Image.open(path_to_images_folder / file_name)
    mask_path = path_to_masks_folder / file_name
    try:
        mask = Image.open(mask_path.with_suffix('.png'))
    except FileNotFoundError:
        try:
            mask = Image.open(mask_path.with_suffix('.jpg'))
        except FileNotFoundError:
            mask = Image.new('RGB', image.size, (0, 0, 0))
    mask = mask.convert('RGB')

    image = image_to_tensor(image)
    mask = image_to_tensor(mask)
    return image, mask


def image_to_tensor(image):
    tensor = torch.clamp(pil_to_tensor(image).float() / 255., 0, 1)
    tensor = tensor.unsqueeze(0)
    tensor = tensor.permute(0, 2, 3, 1)
    return tensor
