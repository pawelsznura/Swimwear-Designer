import torch
import requests
from PIL import Image
from transformers import ViTFeatureExtractor, AutoTokenizer, VisionEncoderDecoderModel


# vit-gpt2-coco-en
# https://huggingface.co/ydshieh/vit-gpt2-coco-en



loc = "ydshieh/vit-gpt2-coco-en"

feature_extractor = ViTFeatureExtractor.from_pretrained(loc)
tokenizer = AutoTokenizer.from_pretrained(loc)
model = VisionEncoderDecoderModel.from_pretrained(loc)
model.eval()


def predict(image_path):
    with Image.open(image_path) as image:
        pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values

        with torch.no_grad():
            output_ids = model.generate(pixel_values, max_length=16, num_beams=4, return_dict_in_generate=True).sequences

        preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]

        return preds


# # We will verify our results on an image of cute cats
# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# with Image.open("insp_img/sunflower.jpg") as image:
#     preds = predict(image)

# print(preds)
# # should produce
# # ['a cat laying on top of a couch next to another cat']

