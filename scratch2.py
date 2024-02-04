

import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests
from PlaceRec.Training.models.backbones import vit_small


url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
model = AutoModel.from_pretrained('facebook/dinov2-base')

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
last_hidden_states = outputs[0]
print(last_hidden_states.shape)

vit_model = vit_small()
out = vit_model(torch.randn(1, 3, 224, 224))
print(out.shape)

model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
out = model.get_intermediate_layers(torch.randn(1, 3, 224, 224))
print(out[0].shape)
