import os
import pickle
from os.path import join
from typing import List, Tuple

import numpy as np
import sklearn
import torch
import torch.nn as nn
import torch.nn.init as init
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from torchvision import transforms
from tqdm import tqdm

from ..utils import get_config, s3_bucket_download
from .base_method import BaseModelWrapper

package_directory = os.path.dirname(os.path.abspath(__file__))
config = get_config()


class AmosNetModel(nn.Module):
    def __init__(self):
        super(AmosNetModel, self).__init__()

        # Conv1
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0)
        init.normal_(self.conv1.weight, std=0.01)
        init.constant_(self.conv1.bias, 0)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.norm1 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75)

        # Conv2
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2)
        init.normal_(self.conv2.weight, std=0.01)
        init.constant_(self.conv2.bias, 1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.norm2 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75)

        # Conv3
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        init.normal_(self.conv3.weight, std=0.01)
        init.constant_(self.conv3.bias, 0)
        self.relu3 = nn.ReLU(inplace=True)

        # Conv4
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2)
        init.normal_(self.conv4.weight, std=0.01)
        init.constant_(self.conv4.bias, 1)
        self.relu4 = nn.ReLU(inplace=True)

        # Conv5
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2)
        init.normal_(self.conv5.weight, std=0.01)
        init.constant_(self.conv5.bias, 1)
        self.relu5 = nn.ReLU(inplace=True)

        # Conv6
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1, groups=2)
        init.normal_(self.conv6.weight, std=0.01)
        init.constant_(self.conv6.bias, 1)
        self.relu6 = nn.ReLU(inplace=True)
        self.pool6 = nn.MaxPool2d(kernel_size=3, stride=2)

        # FC7
        self.fc7_new = nn.Linear(
            256 * 6 * 6, 4096
        )  # Assuming the spatial size is 6x6 after the pooling layers
        init.normal_(self.fc7_new.weight, std=0.005)
        init.constant_(self.fc7_new.bias, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout(p=0.5)

        # FC8
        self.fc8_new = nn.Linear(4096, 2543)
        init.normal_(self.fc8_new.weight, std=0.01)
        init.constant_(self.fc8_new.bias, 0)
        self.prob = nn.Softmax(dim=1)

        # spatial pooling
        self.spatpool4 = nn.AdaptiveMaxPool2d((4, 4))
        self.spatpool3 = nn.AdaptiveMaxPool2d((3, 3))
        self.spatpool2 = nn.AdaptiveMaxPool2d((2, 2))
        self.spatpool1 = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, x):
        # Define the forward pass based on the layers and activations
        x = self.norm1(self.pool1(self.relu1(self.conv1(x))))
        x = self.norm2(self.pool2(self.relu2(self.conv2(x))))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.conv5(x)

        # implement spatial pooling
        feat4 = self.spatpool4(x)
        feat3 = self.spatpool3(x)
        feat2 = self.spatpool2(x)
        feat1 = self.spatpool1(x)

        # flatten conv blocks over height and width
        feat4 = feat4.view(feat4.size(0), feat4.size(1), -1)
        feat3 = feat3.view(feat3.size(0), feat3.size(1), -1)
        feat2 = feat2.view(feat2.size(0), feat2.size(1), -1)
        feat1 = feat1.view(feat1.size(0), feat1.size(1), -1)

        # concatenate channel wise
        feat = torch.cat((feat4, feat3, feat2, feat1), dim=2)
        feat = feat.view(feat.size(0), -1)
        return feat


class SubtractMean:
    def __init__(self, mean_image=None):
        self.mean_image = mean_image

    def __call__(self, image):
        return image - self.mean_image

    def __repr__(self):
        return self.__class__.__name__


class ChannelSwap:
    def __call__(self, tensor):
        """
        Swap channels from RGB to BGR or vice versa.

        Args:
        - tensor (torch.Tensor): Input tensor in CxHxW format.

        Returns:
        - torch.Tensor: Tensor with swapped channels.
        """
        # Swap channels
        return tensor[[2, 1, 0], :, :]


scale_transform = transforms.Lambda(lambda x: x * 255.0)


######################################### AMOSNET #######################################################
model = AmosNetModel()

try:
    mean_image = torch.Tensor(
        np.load(join(config["weights_directory"], "amosnet_mean.npy"))
    )
except:
    mean_image = 0


if isinstance(mean_image, torch.Tensor):
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            scale_transform,
            transforms.Resize((256, 256), antialias=True),
            SubtractMean(mean_image=mean_image),
            ChannelSwap(),
            transforms.Resize((227, 227), antialias=True),
        ]
    )
else:
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize((227, 227), antialias=True),
        ]
    )


class AmosNet(BaseModelWrapper):
    def __init__(self, pretrained: bool = False):
        if pretrained:
            if os.path.exists(config["weights_directory"]):
                model.load_state_dict(
                    torch.load(
                        join(config["weights_directory"], "AmosNet.caffemodel.pt")
                    )
                )
            else:
                raise Exception(
                    f'Could not find weights at {config["weights_directory"]}'
                )

        self.device = "cpu"
        model.to("cpu")

        super().__init__(model=model, preprocess=preprocess, name="amosnet")
        # some layers not implemented on metal
        # some layers not implemented on metal
        if self.device == "mps":
            self.device = "cpu"
        self.model.to(self.device)

    def set_device(self, device: str) -> None:
        if "mps" in device:
            device = "cpu"
        self.device = device
        self.model.to(device)
