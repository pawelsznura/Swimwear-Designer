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

import Stable_Diffusion.txt2img_gpu as onnx


# IMG 2 TXT

# IMG Classifiers 

# import img2text_models.microsoft_beit as microsoft_beit
microsoft_beit = "https://api-inference.huggingface.co/models/microsoft/beit-base-patch16-224-pt22k-ft22k"
# import img2text_models.microsoft_resnet as microsoft_resnet
microsoft_resnet = "https://api-inference.huggingface.co/models/microsoft/resnet-50"
# import img2text_models.microsoft_swin as microsoft_swin
microsoft_swin = "https://api-inference.huggingface.co/models/microsoft/swin-base-patch4-window7-224-in22k"

# import img2text_models.google_vit as google_vit
google_vit = "https://api-inference.huggingface.co/models/google/vit-base-patch16-224"

# import img2text_models.facebook_convnext as facebook_convnext
facebook_convnext = "https://api-inference.huggingface.co/models/facebook/convnext-base-224"
# import img2text_models.facebook_regnet as facebook_regnet
facebook_regnet = "https://api-inference.huggingface.co/models/facebook/regnet-y-008"
# import img2text_models.nvidia_mit as nvidia_mit
nvidia_mit = "https://api-inference.huggingface.co/models/nvidia/mit-b0"

#API key for models 
# headers = {"Authorization": f"Bearer %s" %config.api_img2txt}

models = [microsoft_beit, microsoft_resnet, microsoft_swin,
        google_vit,
        facebook_convnext, facebook_regnet,
        nvidia_mit]

# IMG captioning 

# vit-gpt2-coco-en
# https://huggingface.co/ydshieh/vit-gpt2-coco-en

import img2text_models.vit_gpt2_coco_en as vit_gpt2_coco_en

# vit-gpt2-image-captioning
# https://huggingface.co/nlpconnect/vit-gpt2-image-captioning

import img2text_models.vit_gpt2_image_captioning as vit_gpt2_image_captioning





def generate(img_path, model1, model2, prompt_part, negative_prompts):
    """ img_path - inpiration image
        model1 - img2txt
        model2 - txt2img
        prompt_part - the firt part of the prompt"""

    # SOURCE IMAGE PATH
    # img = "insp_img/apple.jpg"
    img = img_path


    # IMAGE TO TEXT

    # text_despription  = vit_gpt2_coco_en.predict(img)
    # print("vit_gpt2_coco_en")
    # print(text_despription)

    text_despription  = vit_gpt2_image_captioning.predict(img)
    # img_cap_model = "vit_gpt2_image_captioning"
    img_cap_model = model1
    print(img_cap_model)
    print(text_despription)
    # responses = get_text_classification_responses(img)

    # print_text_classification_responses(responses)

    # print(get_best_classification(responses))

    # TEXT TO IMAGE 
    model_id = model2

    # model_id = "runwayml/stable-diffusion-v1-5"
    # # pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    # pipe = StableDiffusionPipeline.from_pretrained(model_id, guidance_scale=7.5, num_inference_steps=15)

    # model_id = "stabilityai/stable-diffusion-2-1"
    pipe = DiffusionPipeline.from_pretrained(model_id)
    pipe = pipe.to("cuda")
    
    # model_id = "stable_diffusion_onnx"    
    # pipe = onnx.onnxPipeline()

    # prompt = "a professional photograph of a design of a two-piece swimsuit inspired by " + text_despription[0]
    prompt = prompt_part + text_despription[0]
    print(prompt)
    neg_prompt = negative_prompts
    print(neg_prompt)

    # image = onnx.onnxImage(width=256, height=256, prompt = prompt, negative_prompt=neg_prompt)


    image = pipe(prompt, negative_prompt=neg_prompt).images[0]  

    # save input img, output img, prompt, model txt2img, model img2txt 

    new_img_path = get_new_file_name("created_images/") + ".png"
    image.save(new_img_path)

    new_txt_path = get_new_file_name("created_images_text/") + ".txt"

    image_details = '''\
input image: {input_image}
output image: {output_image}
prompt: {prompt}
model img2txt: {model_img2txt}
model txt2img: {model_txt2img}\
        '''.format(input_image = img, output_image = new_img_path, 
        prompt = prompt, model_img2txt = img_cap_model, model_txt2img = model_id )



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


def print_text_classification_responses(responses):
    for model, response in zip(models, responses):
        print(model)
        print(json.dumps(response, indent=4))


def get_text_classification_responses(img):
    responses=[]
    retry = True
    while retry:
        retry = False
        for model in models:
            content, status_code = query(img, model)    
            # if status code 503 than retry 
            if status_code == 503:
                retry = True
            else:
                responses.append(content)
        if retry:
            print("models loading, wait 20s")
            time.sleep(20)

    return responses

def query(filename, API_URL):
    with open(filename, "rb") as f:
        # data = {"inputs": f.read(), "options": {"wait_for_model": True}}
        data = f.read()
    response = requests.request("POST", API_URL, headers=headers, data=data)
    # print(response)
    # if response.status_code == 503:
    #     print("retry, sleep 20s")
    #     time.sleep(20)
    #     response = requests.request("POST", API_URL, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8")), response.status_code


def get_best_classification(responses):
    tokens = []
    for model in responses:

        for x in model:
            # print("reponses x")
            # print(x)
            if x['score'] > 0.5:
                # print(x['label'].split(", "))
                # use extend, not append 
                # append method does not work always, 
                # some labels are more than one word and 
                # this would create the issue with appending 
                # a list to the tokens list instead of every element 
                tokens.extend(x['label'].split(", "))
                # tokens.append(x['label'].split(", "))

    # print(tokens)
    return np.unique(tokens)






# if __name__ == "__main__":
#     main()