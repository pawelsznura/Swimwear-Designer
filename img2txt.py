from transformers import BeitFeatureExtractor, BeitForImageClassification
from PIL import Image
import requests
import glob

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

def predict_class_labels(image):
    feature_extractor = BeitFeatureExtractor.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')
    model = BeitForImageClassification.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    # model predicts one of the 21,841 ImageNet-22k classes
    predicted_class_idx = logits.argmax(-1).item()
    return(model.config.id2label[predicted_class_idx])


image_list = []
for filename in glob.glob('insp_img/*.jpg'): 
    print(filename)
    im=Image.open(filename)
    image_list.append(im)


print(predict_class_labels(image_list[2]))
# print(image_list[0])

