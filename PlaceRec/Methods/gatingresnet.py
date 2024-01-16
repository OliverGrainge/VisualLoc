import torchvision.models as models
from PlaceRec.Methods import ResNet50ConvAP
import torch

method = ResNet50ConvAP(pretrained=False)
model = method.model.cpu()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PlaceRec.Methods import BaseModelWrapper

from PlaceRec.Methods import ResNet50ConvAP

base_method = ResNet50ConvAP()
base_model = base_method.model


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionBlock, self).__init__()
        # 1x1 Convolution
        self.path0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 1x1 Convolution followed by 3x3 Convolution
        self.path1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//2, kernel_size=3, dilation=1, padding=1),
            nn.BatchNorm2d(out_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels//2, out_channels//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels//2),
            nn.ReLU(inplace=True)
        )
        # 1x1 Convolution followed by 5x5 Convolution
        self.path2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, kernel_size=1, dilation=1, padding=1),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels//4, out_channels//4, kernel_size=5, padding=1),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(inplace=True)

        )
        # pool path
        self.path3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, kernel_size=1, dilation=4, padding=1),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels//4, out_channels//4, kernel_size=5, padding=1),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(inplace=True)
        )

        self.reduce_conv = nn.Conv2d(int(out_channels * 2), out_channels, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear1 = nn.Linear(int(out_channels*2), int(out_channels*4))
        self.linear2 = nn.Linear(int(out_channels*4), out_channels)
        self.softmax = nn.Softmax(dim=1)
        self.bn = nn.BatchNorm2d(out_channels)


    def forward(self, x):
        # 1x1 Convolution
        b, c, h, w = x.shape
        out0 = self.path0(x)
        out1 = self.path1(x)
        out2 = self.path2(x)
        out3 = self.path3(x)
        fused_features = torch.cat([out0, out1, out2, out3], dim=1)
        fused_features = F.relu(fused_features)

        att = self.pool(fused_features).view(b, -1)
        att = F.relu(self.linear1(att))
        att = self.linear2(att)
        #att = self.softmax(att)

        features = F.relu(self.reduce_conv(fused_features))

        weighted_features = features * att.unsqueeze(-1).unsqueeze(-1)
        features = weighted_features + x
        features = F.relu(self.bn(features))
        return features




class GatingResNetModel(torch.nn.Module):
    def __init__(self, original_model, freeze_old_layers=False):
        super(GatingResNetModel, self).__init__()
        # Copy layers from original model up to the truncation point
        self.conv1 = original_model.backbone.model.conv1
        self.bn1 = original_model.backbone.model.bn1
        self.relu = original_model.backbone.model.relu
        self.maxpool = original_model.backbone.model.maxpool
        self.layer1 = original_model.backbone.model.layer1
        self.layer2 = original_model.backbone.model.layer2
        self.layer3 = original_model.backbone.model.layer3
        #self.layer4 = original_model.backbone.model.layer4
        self.gatinginception = InceptionBlock(1024, 1024)
        #self.conv_last = nn.Conv2d(2048, 1024, 3)
        #self.bn_last = nn.BatchNorm2d(1024)
        self.AAP = nn.AdaptiveAvgPool2d(2)

        if freeze_old_layers: 
            for param in self.conv1.parameters():
                param.requries_grad = False

            for param in self.bn1.parameters():
                param.requries_grad = False

            for param in self.layer1.parameters():
                param.requries_grad = False
            
            for param in self.layer2.parameters():
                param.requries_grad = False
            
            for param in self.layer3.parameters():
                param.requries_grad = False

        else: 
            for param in self.conv1.parameters():
                param.requries_grad = False

            for param in self.bn1.parameters():
                param.requries_grad = False



    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        #x = self.layer4(x)
        x = self.gatinginception(x)
        x = self.AAP(x)
        #x = self.conv_last(x)
        #x = self.bn_last(x)
        x = F.normalize(x.flatten(1), p=2, dim=1)
        return x

preprocess = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((480, 640), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)



class GatingResNet(BaseModelWrapper):
    def __init__(self, pretrained: bool = True):
        model = GatingResNetModel(base_model)
        super().__init__(model=model, preprocess=preprocess, name="gatingresnet")

        self.set_device(self.device)

    def set_device(self, device: str) -> None:
        if "mps" in device:
            device = "cpu"
        self.device = device
        self.model.to(device)
