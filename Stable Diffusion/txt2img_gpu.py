from diffusers import OnnxStableDiffusionPipeline
height=256
width=556
num_inference_steps=50
guidance_scale=7.5
prompt = "hyper realistic desing of a two-pyhtpiece swimwear inspired by a red apple siting on top of a white surface"
negative_prompt=""
pipe = OnnxStableDiffusionPipeline.from_pretrained("./stable_diffusion_onnx", provider="DmlExecutionProvider", safety_checker=None)
image = pipe(prompt, height, width, num_inference_steps, guidance_scale, negative_prompt).images[0] 
image.save("apple_swimwear.png")