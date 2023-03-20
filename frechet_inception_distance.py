import torch
# _ = torch.manual_seed(123)
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms import ToTensor
import os 
from PIL import Image


def frechet_inception_distance(gen_img_paths, real_img_paths):
    fid = FrechetInceptionDistance(feature=64)
    # feature - an integer will indicate the inceptionv3 
    # feature layer to choose. Can be one of the 
    # following: 64, 192, 768, 2048


def transform_to_tensor(img_paths):
    """takes as input array of image paths, returns array of tensor"""
    transform = ToTensor()

    tensors = []

    for img_path in img_paths:
        img = Image.open(img_path)
        img_tensor = transform(img)
        tensors.append(img_tensor)

    return tensors

def get_file_paths(directory):
    file_paths = []
    for file in os.listdir(directory):
        file_paths.append(directory+"/"+file)

    # custom sorting function to sort the list based on the numerical value of the file names
    file_paths.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
    
    # print(file_paths)
    return file_paths

img_paths = get_file_paths("created_images")



