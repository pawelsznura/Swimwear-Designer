import time
import torch
from PIL import Image
from diffusers import OnnxStableDiffusionImg2ImgPipeline

init_image = Image.open("../insp_img/apple_small.jpg")

prompt = "a design of a two-piece swimsuit"

pipe = OnnxStableDiffusionImg2ImgPipeline.from_pretrained("./stable_diffusion_onnx", provider="DmlExecutionProvider", revision="onnx", safety_checker=None)
image = pipe(prompt=prompt, init_image=init_image, strength=0.75, guidance_scale=7.5).images[0] 
image.save("test-output.jpg")