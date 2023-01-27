from diffusers import OnnxStableDiffusionPipeline
height=512
width=512
num_inference_steps=50
guidance_scale=7.5
prompt = "design two-piece swimwear or swimsuit inspired by a field with a bunch of yellow flowers in it"
negative_prompt=""
pipe = OnnxStableDiffusionPipeline.from_pretrained("./stable_diffusion_onnx", provider="DmlExecutionProvider", safety_checker=None)
image = pipe(prompt, height, width, num_inference_steps, guidance_scale, negative_prompt).images[0] 
image.save("sunflower_swimwear_or_swimwear.png")