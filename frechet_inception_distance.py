import torch
_ = torch.manual_seed(123)
from torchmetrics.image.fid import FrechetInceptionDistance
import torchvision.transforms as transforms
from torchvision.transforms import Resize
import os 
from PIL import Image


def frechet_inception_distance(gen, real):
    fid = FrechetInceptionDistance(feature=64)
    # feature - an integer will indicate the inceptionv3 
    # feature layer to choose. Can be one of the 
    # following: 64, 192, 768, 2048
    fid.update(real, real=True)
    fid.update(gen, real=False)
    return fid.compute()


def transform_to_tensor(img_paths):
    """takes as input array of image paths, returns array of tensor"""
    # transform = ToTensor()

    transform = transforms.Compose([
    transforms.PILToTensor()
    ])
    resize = Resize((512, 512))

    tensors = []

    for img_path in img_paths:
        img = Image.open(img_path)
        img_resized = resize(img)
        img_tensor = transform(img_resized)
        # img_tensor = img_tensor.to(torch.uint8)
        tensors.append(img_tensor)

    return torch.stack(tensors)

def get_file_paths(directory):
    file_paths = []
    for file in os.listdir(directory):
        file_paths.append(directory+"/"+file)

    if directory == "created_images":
        # custom sorting function to sort the list based on the numerical value of the file names
        file_paths.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
    
    # print(file_paths)
    return file_paths

gen_img_paths = get_file_paths("created_images")[100:]
real_img_paths = get_file_paths("imagenet")
# print(real_img_paths[-10:])
print(gen_img_paths[:1], gen_img_paths[-1:])
number_of_img = len(gen_img_paths)
print("number of images: "+str(number_of_img))
# print("number of real images: "+str(len(real_img_paths[-number_of_img:])))
gen_tensor = transform_to_tensor(gen_img_paths)

# use the element from end of the list
real_tensor = transform_to_tensor(real_img_paths[-number_of_img:])
# print("\nREAL: \n\n")
# print(real_tensor[:1])

print(frechet_inception_distance(gen_tensor, real_tensor))




