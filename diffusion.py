import base64
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
import torch
from pipeline import get_stable_diffusion_pipeline, get_stable_diffusion_img2img_pipeline, get_stable_diffusion_controlnet_pipeline


def txt2img(
        model: str,
        seed: int,
        prompt: str,
        negative_prompt: str,
        height: int,
        width: int,
        num_inference_steps: int,
        guidance_scale: float,
) -> BytesIO:
    pipe = get_stable_diffusion_pipeline(model)

    generator = torch.manual_seed(seed)
    pil_image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        generator=generator,
        num_images_per_prompt=1,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    ).images[0]

    return convert_image_bytes(pil_image)

def img2img(
        model: str,
        seed: int,
        prompt: str,
        negative_prompt: str,
        strength: float,
        num_inference_steps: int,
        guidance_scale: float,
        image_bytes_base64: str
) -> BytesIO:
    pipe = get_stable_diffusion_img2img_pipeline(model)

    parent_image = convert_image(image_bytes_base64)

    generator = torch.manual_seed(seed)
    pil_image =  pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        generator=generator,
        image=parent_image,
        num_images_per_prompt=1,
        strength=strength,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    ).images[0]

    return convert_image_bytes(pil_image)

def controlnet(
        model: str,
        controlnet_model: str,
        seed: int,
        prompt: str,
        negative_prompt: str,
        controlnet_conditioning_scale: float,
        num_inference_steps: int,
        guidance_scale: float,
        image_bytes_base64: str
) -> BytesIO:
    pipe = get_stable_diffusion_controlnet_pipeline(model, controlnet_model)

    parent_image = convert_canny_image(image_bytes_base64)

    generator = torch.manual_seed(seed)
    pil_image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        generator=generator,
        image=parent_image,
        num_images_per_prompt=1,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    ).images[0]

    return convert_image_bytes(pil_image)

def convert_image(image_bytes_base64: str) -> Image.Image:
    image = Image.open(BytesIO(base64.b64decode(image_bytes_base64.encode("utf-8"))))
    width, height = image.size
    return image.convert("RGB").resize((width, height))

def convert_canny_image(image_bytes_base64: str) -> Image.Image:
    image = convert_image(image_bytes_base64)
    image = np.array(image)
    low_threshold = 100
    high_threshold = 200
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)

    return canny_image

def convert_image_bytes(image: Image.Image) -> BytesIO:
    image_bytes = BytesIO()
    image.save(image_bytes, format="PNG")
    image_bytes.seek(0)

    return image_bytes