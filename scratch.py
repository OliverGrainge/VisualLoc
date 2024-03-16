import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import timm


image = np.random.randint(0, 255, (255, 255, 3)).astype(np.uint8)
image = Image.fromarray(image)



preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to 224x224
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
    ])

img = preprocess(image)

model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
tokens = model(img[None, :])
print(tokens.shape)