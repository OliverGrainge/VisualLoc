from transformers import ViTModel
import os 
from torchvision import transforms
import torch
from PlaceRec.Methods import BaseModelWrapper
import torch.nn as nn

filepath = os.path.dirname(os.path.abspath(__file__))

class vit_base_patch16_224_cls(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

    def forward(self, x):
        return self.vit_model(x).last_hidden_state[:, 0, :]



preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(224, antialias=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class ViT_base_patch16_224_cls(BaseModelWrapper):
    def __init__(self, pretrained: bool = True):
        model = vit_base_patch16_224_cls()
        if pretrained:
            model.load_state_dict(torch.load(os.path.join(filepath, "weights", "msls_vit_tr10_cls.pth")))

        super().__init__(model=model, preprocess=preprocess, name="vit_base_patch16_224_cls")

        self.model.to(self.device)
        self.model.eval()