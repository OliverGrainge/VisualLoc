import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models, transforms


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1.0 / self.p).view(x.shape[0], -1)


backbone = models.resnet18(pretrained=True)
for name, child in backbone.named_children():
    # Freeze layers before conv_3
    if name == "layer3":
        break
    for params in child.parameters():
        params.requires_grad = False

layers = list(backbone.children())[:-2]

backbone = torch.nn.Sequential(*layers)

img = torch.randn(1, 3, 224, 224)
out = backbone(img)
print(out.shape)


def resnet18_gem(descriptor_size=1024):
    backbone = models.resnet18(pretrained=True)
    for name, child in backbone.named_children():
        # Freeze layers before conv_3
        if name == "layer3":
            break
        for params in child.parameters():
            params.requires_grad = False

    layers = list(backbone.children())[:-2]

    backbone = torch.nn.Sequential(*layers)

    return torch.nn.Sequential(backbone, nn.Conv2d(512, descriptor_size, 1, stride=1), GeM())


model = resnet18_gem()
out = model(img)
print(out.shape)
