import img2text_models.microsoft_beit as microsoft_beit
import img2text_models.microsoft_resnet as microsoft_resnet
import img2text_models.microsoft_swin as microsoft_swin

import img2text_models.google_vit as google_vit

import img2text_models.facebook_convnext as facebook_convnext
import img2text_models.facebook_regnet as facebook_regnet

import img2text_models.nvidia_mit as nvidia_mit

import json

import numpy as np

models = [microsoft_beit, microsoft_resnet, microsoft_swin,
        google_vit,
        facebook_convnext, facebook_regnet,
        nvidia_mit]


def main():

    img = "insp_img/bonesaw.jpg"

    responses = get_text_classification_responses(img)

    print_text_classification_responses(responses)

    print(get_best_classification(responses))
    


def print_text_classification_responses(responses):
    for model, response in zip(models, responses):
        print(model.__name__)
        print(json.dumps(response, indent=4))


def get_text_classification_responses(img):
    responses=[]
    for model in models:
        x = model.query(img)    
        # if error 503 than retry 

        responses.append(x)

    return responses

def get_best_classification(responses):
    tokens = []
    for model in responses:

        for x in model:
            # print("reponses x")
            # print(x)
            if x['score'] > 0.5:
                tokens.append(x['label'].split(", "))

    return np.unique(tokens)






if __name__ == "__main__":
    main()