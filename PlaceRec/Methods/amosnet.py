import torch.nn as nn
import torch.nn.init as init
import torch
from .base_method import BaseFunctionality
from torchvision import transforms
import pickle
import os
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import sklearn
from typing import Tuple, List
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import numpy as np
from ..utils import s3_bucket_download


package_directory = os.path.dirname(os.path.abspath(__file__))


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


class AmosNet(BaseFunctionality):
    def __init__(self):
        super().__init__()
        if not os.path.exists(package_directory + "/weights/AmosNet.caffemodel.pt"):
            s3_bucket_download("placerecdata/weights/AmosNet.caffemodel.pt",
                                package_directory + "/AmosNet.caffemodel.pt")

        self.model = AmosNetModel()
        self.model.load_state_dict(
            torch.load(package_directory + "/weights/AmosNet.caffemodel.pt")
        )
        self.model.to(self.device)
        self.model.eval()

        self.mean_image = torch.Tensor(
            np.load(package_directory + "/weights/amosnet_mean.npy")
        )

        self.preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                scale_transform,
                transforms.Resize((256, 256), antialias=True),
                SubtractMean(mean_image=self.mean_image),
                ChannelSwap(),
                transforms.Resize((227, 227), antialias=True),
            ]
        )

        self.map = None
        self.map_desc = None
        self.query_desc = None
        self.name = "amosnet"

    def compute_query_desc(
        self,
        images: torch.Tensor = None,
        dataloader: torch.utils.data.dataloader.DataLoader = None,
        pbar: bool = True,
    ) -> dict:
        if images is not None and dataloader is None:
            all_desc = self.model(images.to(self.device)).detach().cpu().numpy()
        elif dataloader is not None and images is None:
            all_desc = []
            for batch in tqdm(
                dataloader, desc="Computing AmosNet Query Desc", disable=not pbar
            ):
                all_desc.append(
                    self.model(batch.to(self.device)).detach().cpu().numpy()
                )
            all_desc = np.vstack(all_desc)

        query_desc = {"query_descriptors": all_desc}
        self.set_query(query_desc)
        return query_desc

    def compute_map_desc(
        self,
        images: torch.Tensor = None,
        dataloader: torch.utils.data.dataloader.DataLoader = None,
        pbar: bool = True,
    ) -> dict:
        if images is not None and dataloader is None:
            all_desc = self.model(images.to(self.device)).detach().cpu().numpy()
        elif dataloader is not None and images is None:
            all_desc = []
            for batch in tqdm(
                dataloader, desc="Computing AmosNet Map Desc", disable=not pbar
            ):
                all_desc.append(
                    self.model(batch.to(self.device)).detach().cpu().numpy()
                )
            all_desc = np.vstack(all_desc)
        else:
            raise Exception("can only pass 'images' or 'dataloader'")

        map_desc = {"map_descriptors": all_desc}
        self.set_map(map_desc)
        return map_desc
