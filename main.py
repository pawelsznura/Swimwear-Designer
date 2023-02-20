import json
import numpy as np
import config
import requests
import time
import os

from diffusers import StableDiffusionPipeline, DiffusionPipeline
import torch

from PIL import Image
from transformers import ViTFeatureExtractor, AutoTokenizer, VisionEncoderDecoderModel



# IMG 2 TXT

# IMG captioning 

# vit-gpt2-coco-en
# https://huggingface.co/ydshieh/vit-gpt2-coco-en
import img2text_models.vit_gpt2_coco_en as vit_gpt2_coco_en

# vit-gpt2-image-captioning
# https://huggingface.co/nlpconnect/vit-gpt2-image-captioning
import img2text_models.vit_gpt2_image_captioning as vit_gpt2_image_captioning


def generate(img_path, model1, model2, prompt_part, prompt_end, negative_prompts):
    """ img_path - inspiration image
        model1 - img2txt
        model2 - txt2img
        prompt_part - the first part of the prompt
        prompt_end - last part of prompt 
        negative_prompts - negative prompt"""

    # measuring the time of whole process
    start_time = time.time()

    # SOURCE IMAGE PATH
    # img = "insp_img/apple.jpg"
    img = img_path


    # IMAGE TO TEXT
    img_cap_model = model1

    if img_cap_model == "vit_gpt2_image_captioning":
        text_description  = vit_gpt2_image_captioning.predict(img)
        print(img_cap_model)
        print(text_description)
    elif img_cap_model == "vit_gpt2_coco_en":
        text_description  = vit_gpt2_coco_en.predict(img)
        print(img_cap_model)
        print(text_description)

    # PROMPT

    prompt = prompt_part + text_description[0] + prompt_end
    print(prompt)
    neg_prompt = negative_prompts
    print(neg_prompt)
    

    # TEXT TO IMAGE 
    model_id = model2

    # model_id = "runwayml/stable-diffusion-v1-5"
    # # pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    # pipe = StableDiffusionPipeline.from_pretrained(model_id, guidance_scale=7.5, num_inference_steps=15)

    # model_id = "stabilityai/stable-diffusion-2-1"
    pipe = DiffusionPipeline.from_pretrained(model_id)
    # could be used if a cuda supported GPU is available 
    # pipe = pipe.to("cuda")


    image = pipe(prompt, negative_prompt=neg_prompt).images[0]  

    # save input img, output img, prompt, model txt2img, model img2txt 

    new_img_path = get_new_file_name("created_images/") + ".png"
    image.save(new_img_path)

    new_txt_path = get_new_file_name("created_images_text/") + ".txt"

    end_time = time.time()

    total_time = end_time - start_time

    total_time_minutes = total_time / 60

    image_details = '''\
input image: {input_image}
output image: {output_image}
prompt: {prompt}
model img2txt: {model_img2txt}
model txt2img: {model_txt2img}
execution time in minutes: {total_minutes}\
        '''.format(input_image = img, output_image = new_img_path, 
        prompt = prompt, model_img2txt = img_cap_model, model_txt2img = model_id,
         total_minutes = total_time_minutes )



    with open(new_txt_path, "w") as text_file:
        text_file.write(image_details)


    
def get_new_file_name(dir_path):
    

    # TODO what if directory is empty? 


    # FIXED not sure if the list is in correct order (possible error 10.png cos 1 < 9)
    # last = os.listdir(dir_path)[len(os.listdir(dir_path))-1]
    sorted_list = sorted(os.listdir(dir_path), key=lambda x: int(x.split(".")[0]))
    last = sorted_list[-1]

    # print(os.listdir(dir_path))
    # print(sorted_list)
    # print("last element " + last)

    img_number = last.split(".")[0]
    # print("number = "+img_number)

    new_number = int(img_number) + 1

    return dir_path + str(new_number) 



# if __name__ == "__main__":
#     main()