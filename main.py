from fastapi import FastAPI
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse

import diffusion

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/hello")
def hello():
    return {"message": "Hello!"}

class Txt2Img(BaseModel):
    seed: int
    prompt: str
    negative_prompt: str
    model: str
    num_inference_steps: int
    guidance_scale: float
    height: int
    width: int


class Img2Img(BaseModel):
    seed: int
    prompt: str
    negative_prompt: str
    model: str
    num_inference_steps: int
    guidance_scale: float
    strength: float
    image_bytes_base64: str

class Controlnet(BaseModel):
    seed: int
    prompt: str
    negative_prompt: str
    model: str
    controlnet_model: str
    num_inference_steps: int
    guidance_scale: float
    controlnet_conditioning_scale: float
    image_bytes_base64: str


@app.post("/txt2img")
async def txt2img(data: Txt2Img):
    image_bytes = diffusion.txt2img(**data.dict())

    return StreamingResponse(image_bytes, media_type="image/png")


@app.post("/img2img")
def img2img(data: Img2Img):
    image_bytes = diffusion.img2img(**data.dict())

    return StreamingResponse(image_bytes, media_type="image/png")

@app.post("/controlnet")
def controlnet(data: Controlnet):
    image_bytes = diffusion.controlnet(**data.dict())

    return StreamingResponse(image_bytes, media_type="image/png")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8080, reload=False, workers=1)