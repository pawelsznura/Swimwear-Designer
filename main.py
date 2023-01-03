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

import json
import numpy as np
import config
import requests
import time

from diffusers import StableDiffusionPipeline
import torch

headers = {"Authorization": f"Bearer %s" %config.api_img2txt}

models = [microsoft_beit, microsoft_resnet, microsoft_swin,
        google_vit,
        facebook_convnext, facebook_regnet,
        nvidia_mit]

# models = [microsoft_beit]


def main():

    img = "insp_img/sunflower.jpg"


    responses = get_text_classification_responses(img)

    print_text_classification_responses(responses)

    print(get_best_classification(responses))

    # model prompt 
    # a design of a *color* swimsuit with the shape of a *shape* shape inspired by *insp* 
    # a design of a silver swimsuit inspired by sunflowers  




    model_id = "runwayml/stable-diffusion-v1-5"
    # pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    # pipe = pipe.to("cpu")

    prompt = "a hyper realistic design of a green two-piece swimsuit inspired by bonesaw"
    image = pipe(prompt).images[0]  
        
    image.save("sss.png")
    


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






if __name__ == "__main__":
    main()