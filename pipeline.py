import os
import _pickle as pickle
from functools import lru_cache

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusionImg2ImgPipeline, StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

@lru_cache()
def get_stable_diffusion_pipeline(model: str) -> StableDiffusionPipeline:
    if model != "anything-v3.0" and model != "Counterfeit-V2.5":
        raise ValueError(f"Unknown model_id: {model}")

    pipe = pickle.load(open(f"pipelines/stable-diffusion/txt2img/{model}.pkl", "rb"))

    return pipe

@lru_cache()
def get_stable_diffusion_img2img_pipeline(model: str) -> StableDiffusionImg2ImgPipeline:
    if model != "anything-v3.0" and model != "Counterfeit-V2.5":
        raise ValueError(f"Unknown model_id: {model}")

    pipe = pickle.load(open(f"pipelines/stable-diffusion/img2img/{model}.pkl", "rb"))

    return pipe

@lru_cache()
def get_stable_diffusion_controlnet_pipeline(model: str, controlnet_model: str) -> StableDiffusionControlNetPipeline:
    if model != "anything-v3.0" and model != "Counterfeit-V2.5" and controlnet_model != "sd-controlnet-canny":
        raise ValueError(f"Unknown model_id: {model} {controlnet_model}")

    pipe = pickle.load(open(f"pipelines/stable-diffusion/controlnet/{model}_{controlnet_model}.pkl", "rb"))

    return pipe

if __name__ == "__main__":
    def stable_diffusion_pipeline(model_path: str):
        pipeline = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
        pipeline.to("cuda")
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

        return pipeline

    def stable_diffusion_img2img_pipeline(model_path: str):
        pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
        pipeline.to("cuda")
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

        return pipeline

    def stable_diffusion_controlnet_pipeline(model_controlnet_path: str, model_path: str):
        controlnet = ControlNetModel.from_pretrained(model_controlnet_path, torch_dtype=torch.float16)
        pipeline = StableDiffusionControlNetPipeline.from_pretrained(model_path, controlnet=controlnet, torch_dtype=torch.float16)
        pipeline.to("cuda")
        pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)

        return pipeline

    def save_pipeline():
        model_paths = ["gsdf/Counterfeit-V2.5", "Linaqruf/anything-v3.0"]
        model_controlnet_paths = ["lllyasviel/sd-controlnet-canny"]
        draw_types = ["txt2img", "img2img", "controlnet"]

        for draw_type in draw_types:
            os.makedirs(f"pipelines/stable-diffusion/{draw_type}", exist_ok=True)
            for model_path in model_paths:
                # /を削除したものがmodel
                model = model_path.split("/")[1]
                if draw_type == "img2img":
                    # 存在する場合はスキップ
                    if os.path.exists(f"pipelines/stable-diffusion/{draw_type}/{model}.pkl"):
                        print(f"Skipped pipeline for {draw_type} {model}.")
                        continue
                    pipeline = stable_diffusion_img2img_pipeline(model_path)
                    pickle.dump(pipeline, open(f"pipelines/stable-diffusion/{draw_type}/{model}.pkl", "wb"))
                    print(f"Saved pipeline for {draw_type} {model}.")
                elif draw_type == "txt2img":
                    # 存在する場合はスキップ
                    if os.path.exists(f"pipelines/stable-diffusion/{draw_type}/{model}.pkl"):
                        print(f"Skipped pipeline for {draw_type} {model}.")
                        continue
                    pipeline = stable_diffusion_pipeline(model_path)
                    pickle.dump(pipeline, open(f"pipelines/stable-diffusion/{draw_type}/{model}.pkl", "wb"))
                    print(f"Saved pipeline for {draw_type} {model}.")
                elif draw_type == "controlnet":
                    for model_controlnet_path in model_controlnet_paths:
                        model_controlnet = model_controlnet_path.split("/")[1]
                        # 存在する場合はスキップ
                        if os.path.exists(f"pipelines/stable-diffusion/{draw_type}/{model}_{model_controlnet}.pkl"):
                            print(f"Skipped pipeline for {draw_type} {model} {model_controlnet}.")
                            continue
                        pipeline = stable_diffusion_controlnet_pipeline(model_controlnet_path, model_path)
                        pickle.dump(pipeline, open(f"pipelines/stable-diffusion/{draw_type}/{model}_{model_controlnet}.pkl", "wb"))
                        print(f"Saved pipeline for {draw_type} {model} {model_controlnet}.")


    save_pipeline()