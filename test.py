# File to do some run some quick code snippets
import main 

# import Stable_Diffusion.txt2img_gpu as onnx


# prompt = "guy sitting in front of computer"



# model_id = "stable_diffusion_onnx"    
# pipe = onnx.onnxPipeline()


# img_path = 'test.jpg'
# neg_prompt = ""
# image = pipe(prompt, negative_prompt=neg_prompt).images[0]  

# image.save(img_path)


# new_dir = main.get_new_file_name("created_images/")
# print(new_dir)


import torch
from diffusers import StableDiffusionAttendAndExcitePipeline

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionAttendAndExcitePipeline.from_pretrained(model_id, torch_dtype=torch.float16)
# pipe = pipe.to("cuda")

prompt = "a cat and a frog"

# use get_indices function to find out indices of the tokens you want to alter
pipe.get_indices(prompt)

token_indices = [2, 5]
seed = 6141
generator = torch.manual_seed(seed)

images = pipe(
    prompt=prompt,
    token_indices=token_indices,
    guidance_scale=7.5,
    generator=generator,
    num_inference_steps=50,
    max_iter_to_alter=25,
).images

image = images[0]
image.save(f"../images/{prompt}_{seed}.png")