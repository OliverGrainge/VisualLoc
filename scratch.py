import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


image = np.random.randint(0, 255, (255, 255, 3)).astype(np.uint8)
image = Image.fromarray(image)

def make_patchable(img_size):
    return (img_size // 14) * 14

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.Lambda(lambda img: transforms.functional.center_crop(img, (make_patchable(img.shape[1]), make_patchable(img.shape[2])))),
])


img = preprocess(image)

print(img.shape)
