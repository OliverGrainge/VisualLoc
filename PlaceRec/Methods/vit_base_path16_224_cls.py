import os
from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision import transforms
from transformers import ViTModel

from PlaceRec.Methods import BaseModelWrapper

filepath = os.path.dirname(os.path.abspath(__file__))


class vit_base_patch16_224_cls(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        encoder_layers = list(self.backbone.encoder.layer.children())
        self.backbone.encoder.layer = nn.ModuleList(encoder_layers[:10])

    def forward(self, x):
        return self.backbone(x).last_hidden_state[:, 0, :]



preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224), antialias=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class ViT_base_patch16_224_cls(BaseModelWrapper):
    def __init__(self, pretrained: bool = True):
        model = vit_base_patch16_224_cls()
        if pretrained:
            state_dict = torch.load(os.path.join(filepath, "weights", "msls_vit_tr10_cls.pth"))["model_state_dict"]
            if list(state_dict.keys())[0].startswith('module'):
                state_dict = OrderedDict({k.replace('module.', ''): v for (k, v) in state_dict.items()})

            for model_key, loaded_key in zip(model.state_dict(), state_dict):
                if model_key in state_dict:
                    model.state_dict()[model_key].copy_(state_dict[loaded_key])
                else:
                    print(f"Key '{model_key}' not found in loaded state_dict. Skipping...")

           
        
            model.load_state_dict(state_dict)

        super().__init__(model=model, preprocess=preprocess, name="vit_base_patch16_224_cls")

        self.model.to(self.device)
        self.model.eval()