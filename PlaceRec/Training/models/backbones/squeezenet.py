import torch.nn as nn
import torchvision.models as models


def SqueezeNet(model_name="squeezenet", pretrained=True, layers_to_freeze=2, layers_to_crop=[]): 
    model = models.squeezenet1_0(pretrained=True)
    if pretrained: 
        for i, child in enumerate(model.features.children()):
            if i <= layers_to_freeze:
                for param in child.parameters():
                    param.requires_grad = False
    return model.features