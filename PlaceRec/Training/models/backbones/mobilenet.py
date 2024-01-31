import torch.nn as nn
import torchvision.models as models
from torchvision.models import MobileNet_V2_Weights

def MobileNetV2(model_name="mobilenet_v2", pretrained=True, layers_to_freeze=2, layers_to_crop=[]): 
    model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    if pretrained: 
        for i, child in enumerate(model.features.children()):
            if i <= layers_to_freeze:
                for param in child.parameters():
                    param.requires_grad = False
    return model.features