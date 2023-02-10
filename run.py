# script for running this app 

import main
import os


all_img = os.listdir("insp_img/")

# print(all_img)

# img = "insp_img/"+all_img[2]
# img = "insp_img/elephant.jpg"

# Prompt engineering 
# how to keep everything in frame? 

details_list = ["Hyper realistic", "Highly detailed", "high resolution", "detail", "8 k", "dslr", "cinematic lighting", "hyperdetailed"]
fashion_list = ["concept", "fashion design", "elegant", "luxury",]

designers_list = ["design by", "balenciaga", "laagam", "loewe", "aimeleondore", "Louis Vuitton"]

front_prompt = ["A sketch of ", "A drawing of", "A photograph of", "good lighting", "ray tracing", "realistic" ]

styles = ["Modernistic", "Abstract"]

neg_prompt2 = "ugly, deformed, deformity, ugliness, blurry, disfigured, poorly drawn face, mutation, mutated, extra limbs, messy drawing, text, cropped head, cropped face"


prompt_part = "female swimsuit design inspired by "
neg_prompt = "ugly, tiling, poorly drawn, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, blurred, text, watermark, grainy"


img_cap_model = "vit_gpt2_image_captioning"

model_id = "runwayml/stable-diffusion-v1-5"
# model_id = "stabilityai/stable-diffusion-2-1"
# model_id = "stable_diffusion_onnx"
    

# run all img 
for img in all_img:
    main.generate("insp_img/"+img, img_cap_model, model_id, prompt_part, neg_prompt)

# main.generate(img, img_cap_model, model_id, prompt_part, neg_prompt)

