from PlaceRec.Training.models import get_model


model = get_model("vgg16", "gem", descriptor_size=4000)

import torch 

x = torch.randn(1, 3, 320, 320)
out = model(x)
print(out.shape)