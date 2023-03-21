import torch
from torchvision.models import inception_v3
from torchvision.transforms import ToTensor
from PIL import Image
import os 

# needs to be improved, looks like something is wrong 
# when running it needs to much memory 

def inception_score(img_paths):
    # Load pre-trained Inception V3 model
    model = inception_v3(pretrained=True)
    model.eval()

    # define image transformation
    transform = ToTensor()

    # Load images from list of paths and calculate probabilities
    probs = []
    for img_path in img_paths:
        img = Image.open(img_path)
        img_tensor = transform(img).unsqueeze(0)
        prob = torch.softmax(model(img_tensor), dim=1)
        probs.append(prob)

    # Calculate inception score
    probs = torch.cat(probs)
    kl_divergence = probs * (torch.log(probs) - torch.log(torch.mean(probs, dim=0)))
    score = torch.exp(torch.mean(torch.sum(kl_divergence, dim=1)))

    return score.item()


def get_file_paths(directory):
    file_paths = []
    for file in os.listdir(directory):
        file_paths.append(directory+"/"+file)

    # custom sorting function to sort the list based on the numerical value of the file names
    file_paths.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
    
    # print(file_paths)
    return file_paths

img_paths = get_file_paths("created_images")
# print(img_paths[-10:])
print(inception_score(img_paths[-200:]))