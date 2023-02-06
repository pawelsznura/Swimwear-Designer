# script for running this app 

import main
import os


all_img = os.listdir("insp_img/")

img = "insp_img/"+all_img[1]

prompt_part = "female swimwear design inspired by "
neg_prompt = "ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, blurred, text, watermark, grainy"


img_cap_model = "vit_gpt2_image_captioning"

model_id = "runwayml/stable-diffusion-v1-5"
# model_id = "stabilityai/stable-diffusion-2-1"
    



main.generate(img, img_cap_model, model_id, prompt_part, neg_prompt)


