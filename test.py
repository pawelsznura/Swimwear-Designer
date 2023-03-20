import torch

x=torch.randint(100, 255, (100, 3, 299, 299), dtype=torch.uint8)

print(type(x))