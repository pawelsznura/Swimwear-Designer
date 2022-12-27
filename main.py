import img2text_models.microsoft_beit as microsoft_beit
import img2text_models.microsoft_resnet as microsoft_resnet
import img2text_models.microsoft_swin as microsoft_swin

import img2text_models.google_vit as google_vit

import img2text_models.facebook_convnext as facebook_convnext
import img2text_models.facebook_regnet as facebook_regnet

import img2text_models.nvidia_mit as nvidia_mit

models = [microsoft_beit, microsoft_resnet, microsoft_swin,
        google_vit,
        facebook_convnext, facebook_regnet,
        nvidia_mit]

responses=[]

def main():

    for model in models:
        x = model.query("insp_img/sponge2.jpg")
        responses.append(x)

    # response  = google_vit.query("insp_img/sponge2.jpg")

    print(responses)




if __name__ == "__main__":
    main()