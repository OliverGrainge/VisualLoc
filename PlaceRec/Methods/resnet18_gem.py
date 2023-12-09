import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models, transforms
from PlaceRec.utils import L2Norm
from torchvision.models import ResNet18_Weights

from .base_method import BaseModelWrapper


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1.0 / self.p).view(x.shape[0], -1)


def resnet18_gem(descriptor_size=1024):
    backbone = models.resnet18(weights=ResNet18_Weights)

    layers = list(backbone.children())[:-2]

    backbone = torch.nn.Sequential(*layers)

    return torch.nn.Sequential(backbone, nn.Conv2d(512, descriptor_size, 1, stride=1), GeM(), L2Norm())


preprocess = transforms.Compose(
    [
        transforms.Resize(256),  # Resize the image to 256x256 pixels
        transforms.CenterCrop(224),  # Crop the image to 224x224 pixels about the center
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Normalize(  # Normalize the image
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]  # These are the ImageNet mean and std values
        ),
    ]
)


class ResNet18GeM(BaseModelWrapper):
    def __init__(self, pretrained: bool = True):
        if pretrained:
            raise NotImplementedError
        else:
            model = resnet18_gem(descriptor_size=2048)

        super().__init__(model=model, preprocess=preprocess, name="resnet18_gem")

        self.model.to(self.device)
        self.model.eval()

if __name__ == "__main__":
    model = resnet18_gem()
    print(model)