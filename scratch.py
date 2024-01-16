import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

model = torch.hub.load(
    "gmberton/eigenplaces", "get_trained_model", backbone="ResNet50", fc_output_dim=2048
)

image = np.random.randint(0, 255, (255, 255, 3)).astype(np.uint8)
image = Image.fromarray(image)


preprocess = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((480, 640), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

img = preprocess(image)

out = model(img[None, :])

print(out.shape, torch.norm(out))
