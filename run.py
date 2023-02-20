# script for running this app 

import main
import os


all_img = os.listdir("insp_img/")

# print(all_img)

# img = "insp_img/"+all_img[2]
img = "insp_img/elephant.jpg"

# Prompt engineering 
# how to keep everything in frame? 

details_list = ["Hyper realistic", "Highly detailed", "high resolution", "detail", "8 k", "dslr", "cinematic lighting", "hyperdetailed"]
fashion_list = ["concept", "fashion design", "elegant", "luxury",]

designers_list = ["design by", "balenciaga", "laagam", "loewe", "aimeleondore", "Louis Vuitton"]

# front_prompt = ["A sketch of a", "A drawing of a", "A photograph of a"]
front_prompt = ["A photograph of a"]

styles = ["Modernistic", "Abstract","realistic"]

swimsuit = ["swimsuit","swimwear", "two-piece","one-piece"]

gender = ["female", "male", "unisex"]

prompt_part = front_prompt[0] +" "+ gender[0] +" "+ swimsuit[3] +" "+ swimsuit[0] + " inspired by "

prompt_end = " highly detailed, 8 k, hyper realistic"

# TODO function to generate prompts
#  front and back 

neg_prompt2 = "ugly, deformed, deformity, ugliness, blurry, disfigured, poorly drawn face, mutation, mutated, extra limbs, messy drawing, text, cropped head, cropped face"


# prompt_part = "female swimsuit design inspired by"
# prompt_end = "highly detailed, 8 k, hyper realistic"
neg_prompt = "ugly, tiling, poorly drawn, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, blurred, text, watermark, grainy"


img_cap_model = "vit_gpt2_image_captioning"

model_id = "runwayml/stable-diffusion-v1-5"
# model_id = "stabilityai/stable-diffusion-2-1"
# model_id = "stable_diffusion_onnx"
    

# run all img 
# for img in all_img:
    # main.generate("insp_img/"+img, img_cap_model, model_id, prompt_part, prompt_end, neg_prompt)

main.generate(img, img_cap_model, model_id, prompt_part, prompt_end, neg_prompt)


# for front in front_prompt:
#     for concept in fashion_list:
#         for style in styles:
#            prompt_part = front +" "+ gender[0] +" "+ swimsuit[3] +" "+ swimsuit[0] + " inspired by " 
#            prompt_end = ", " + concept +", "+ style
#         #    prompt_end = " " + concept 
#            main.generate(img, img_cap_model, model_id, prompt_part, prompt_end, neg_prompt2)

