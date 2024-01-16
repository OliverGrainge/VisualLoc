import torch
import torchvision.models as models
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
from torchvision.models._utils import IntermediateLayerGetter
from PIL import Image
import numpy as np
from torchvision import transforms
from torch import nn
from PlaceRec.Methods import BaseModelWrapper


import torch
import torch.nn as nn
import torch.nn.functional as F

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionBlock, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.branch3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.branch5x5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)

    def forward(self, x, weights):
        branch1x1 = self.branch1x1(x) * weights[:, 0][:, None, None, None]
        branch3x3 = self.branch3x3(x) * weights[:, 1][:, None, None, None]
        branch5x5 = self.branch5x5(x) * weights[:, 2][:, None, None, None]
        fused_outputs = torch.stack([branch1x1, branch3x3, branch5x5]).mean(0)
        return fused_outputs

class InceptionV2Block(nn.Module):
    def __init__(self, in_channels, reduce_channels3x3, reduce_channels5x5, out_channels):
        super(InceptionV2Block, self).__init__()
        # 1x1 Convolution
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # 1x1 Convolution followed by 3x3 Convolution
        self.reduce3x3 = nn.Conv2d(in_channels, reduce_channels3x3, kernel_size=1)
        self.conv3x3 = nn.Conv2d(reduce_channels3x3, out_channels, kernel_size=3, padding=1)
        # 1x1 Convolution followed by 5x5 Convolution
        self.reduce5x5 = nn.Conv2d(in_channels, reduce_channels5x5, kernel_size=1)
        self.conv5x5 = nn.Conv2d(reduce_channels5x5, out_channels, kernel_size=5, padding=2)
        # 3x3 Max Pooling followed by 1x1 Convolution
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv_pool = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, weights):
        # 1x1 Convolution
        out1x1 = F.relu(self.conv1x1(x)) * weights[:, 0][:, None, None, None]
        out3x3 = F.relu(self.conv3x3(self.reduce3x3(x))) * weights[:, 1][:, None, None, None]
        out5x5 = F.relu(self.conv5x5(self.reduce5x5(x))) * weights[:, 2][:, None, None, None]
        out_pool = F.relu(self.conv_pool(self.pool(x))) * weights[:, 3][:, None, None, None]
        fused_outputs = torch.stack([out1x1, out3x3, out5x5, out_pool]).mean(0)
        return fused_outputs

class GatingFunction(nn.Module):
    def __init__(self, in_channels, n_weights):
        super(GatingFunction, self).__init__()
        self.conv = nn.Conv2d(in_channels, n_weights, kernel_size=5)
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.softmax = nn.Softmax(dim=1)
        self.n_weights = n_weights

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.conv(x)
        x = self.pool(x).view(b, self.n_weights)
        x = self.softmax(x)
        return x
    

class GatingInceptionBlock(nn.Module):
    def __init__(self, in_channels, reduce_channels3x3, reduce_channels5x5, out_channels):
        super(GatingInceptionBlock, self).__init__()
        self.inception_module = InceptionV2Block(in_channels=in_channels, 
                                                 reduce_channels3x3=reduce_channels3x3,
                                                 reduce_channels5x5=reduce_channels5x5,
                                                 out_channels=out_channels)
        self.gating_module = GatingFunction(in_channels=in_channels,
                                            n_weights=4)
    
    def forward(self, x):
        fusion_weights = self.gating_module(x)
        fused_features = self.inception_module(x, fusion_weights)
        return fused_features


class GatingInceptionModelLarge(nn.Module):
    def __init__(self, descriptor_size=2048): 
        super(GatingInceptionModelLarge, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv5 = nn.Conv2d(64, 80, kernel_size=1)
        self.conv6 = nn.Conv2d(80, 192, kernel_size=3, stride=2)
        self.conv7 = nn.Conv2d(192, 288, kernel_size=3, stride=1)

        self.inception_block1 = GatingInceptionBlock(288, 128, 128, 288)
        self.inception_block2 = GatingInceptionBlock(288, 128, 128, 288)
        self.inception_block3 = GatingInceptionBlock(288, 128, 128, 768)

        self.inception_block4 = GatingInceptionBlock(768, 256, 256, 768)
        self.inception_block5 = GatingInceptionBlock(768, 256, 256, 768)
        self.inception_block6 = GatingInceptionBlock(768, 256, 256, 768)
        self.inception_block7 = GatingInceptionBlock(768, 256, 256, 768)
        self.inception_block8 = GatingInceptionBlock(768, 256, 256, 1280)

        self.inception_block9 = GatingInceptionBlock(1280, 512, 512, 1280)
        self.inception_block10 = GatingInceptionBlock(1280, 512, 512, 2048)
        self.conv_final = nn.Conv2d(2048, descriptor_size//2, kernel_size=1)
        self.AAP = nn.AdaptiveAvgPool2d((2, 2))
    
    def forward(self, x):
        x= F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool4(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))

        x = F.relu(self.inception_block1(x))
        x = F.relu(self.inception_block2(x))
        x = F.relu(self.inception_block3(x))
        

        x = F.relu(self.inception_block4(x))
        x = F.relu(self.inception_block5(x))
        x = F.relu(self.inception_block6(x))
        x = F.relu(self.inception_block7(x))
        x = F.relu(self.inception_block8(x))

        x = F.relu(self.inception_block9(x))
        x = F.relu(self.inception_block10(x))

        x = F.relu(self.conv_final(x))
        x = self.AAP(x)
        x = F.normalize(x.flatten(1), p=2, dim=1)
        return x


class GatingInceptionModelSmall(nn.Module):
    def __init__(self, descriptor_size=2048):
        super(GatingInceptionModelSmall, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv4 = nn.Conv2d(64, 80, kernel_size=1)
        self.conv5 = nn.Conv2d(80, 192, kernel_size=3, stride=2)
        #self.pool6 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.inception_block1 = GatingInceptionBlock(192, 96, 96, 192)
        self.inception_block2 = GatingInceptionBlock(192, 96, 96, 288)
        self.poolblock1 = nn.MaxPool2d(kernel_size=3, stride=1)

        self.inception_block3 = GatingInceptionBlock(288, 128, 128, 288)
        self.inception_block4 = GatingInceptionBlock(288, 128, 128, 512)
        self.inception_block6 = GatingInceptionBlock(512, 256, 256, 512)
        self.poolblock2 = nn.MaxPool2d(kernel_size=3, stride=1)

        self.inception_block7 = GatingInceptionBlock(512, 256, 256, 1028)
        self.conv_final = nn.Conv2d(1028, descriptor_size//2, kernel_size=1)
        self.AAP = nn.AdaptiveAvgPool2d((2, 2))
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        #x = self.pool6(x)


        x = F.relu(self.inception_block1(x))
        x = F.relu(self.inception_block2(x))
        x = self.poolblock1(x)

        x = F.relu(self.inception_block3(x))
        x = F.relu(self.inception_block4(x))
        x = self.poolblock2(x)

        x = F.relu(self.inception_block6(x))
        x = F.relu(self.inception_block7(x))

        x = F.relu(self.conv_final(x))
        x = self.AAP(x)
        x = F.normalize(x.flatten(1), p=2, dim=1)
        return x



preprocess = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((480, 640), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class GatingInceptionLarge(BaseModelWrapper):
    def __init__(self, pretrained: bool = True):
        model = GatingInceptionModelLarge()
        super().__init__(model=model, preprocess=preprocess, name="gatinginceptionlarge")

        self.set_device(self.device)

    def set_device(self, device: str) -> None:
        if "mps" in device:
            device = "cpu"
        self.device = device
        self.model.to(device)

class GatingInceptionSmall(BaseModelWrapper):
    def __init__(self, pretrained: bool = True):
        model = GatingInceptionModelSmall()
        super().__init__(model=model, preprocess=preprocess, name="gatinginceptionsmall")

        self.set_device(self.device)

    def set_device(self, device: str) -> None:
        if "mps" in device:
            device = "cpu"
        self.device = device
        self.model.to(device)
