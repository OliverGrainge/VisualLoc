import os

from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader
import torch
from torchvision.models import ResNet50_Weights
from PIL import Image
import tqdm
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import pickle
import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from typing import Tuple
from .base_method import BaseTechnique
from sklearn.neighbors import NearestNeighbors
import sklearn

try: 
    import faiss 
except:
    pass

package_directory = os.path.dirname(os.path.abspath(__file__))

class ResNet(nn.Module):
    def __init__(self,
                 model_name='resnet50',
                 pretrained=True,
                 layers_to_freeze=2,
                 layers_to_crop=[],
                 ):
        """Class representing the resnet backbone used in the pipeline
        we consider resnet network as a list of 5 blocks (from 0 to 4),
        layer 0 is the first conv+bn and the other layers (1 to 4) are the rest of the residual blocks
        we don't take into account the global pooling and the last fc

        Args:
            model_name (str, optional): The architecture of the resnet backbone to instanciate. Defaults to 'resnet50'.
            pretrained (bool, optional): Whether pretrained or not. Defaults to True.
            layers_to_freeze (int, optional): The number of residual blocks to freeze (starting from 0) . Defaults to 2.
            layers_to_crop (list, optional): Which residual layers to crop, for example [3,4] will crop the third and fourth res blocks. Defaults to [].

        Raises:
            NotImplementedError: if the model_name corresponds to an unknown architecture. 
        """
        super().__init__()
        self.model_name = model_name.lower()
        self.layers_to_freeze = layers_to_freeze

        if pretrained:
            # the new naming of pretrained weights, you can change to V2 if desired.
            weights = 'IMAGENET1K_V1'
        else:
            weights = None

        if 'swsl' in model_name or 'ssl' in model_name:
            # These are the semi supervised and weakly semi supervised weights from Facebook
            self.model = torch.hub.load(
                'facebookresearch/semi-supervised-ImageNet1K-models', model_name)
        else:
            if 'resnext50' in model_name:
                self.model = torchvision.models.resnext50_32x4d(weights=weights)
            elif 'resnet50' in model_name:
                self.model = torchvision.models.resnet50(weights=weights)
            elif '101' in model_name:
                self.model = torchvision.models.resnet101(weights=weights)
            elif '152' in model_name:
                self.model = torchvision.models.resnet152(weights=weights)
            elif '34' in model_name:
                self.model = torchvision.models.resnet34(weights=weights)
            elif '18' in model_name:
                # self.model = torchvision.models.resnet18(pretrained=False)
                self.model = torchvision.models.resnet18(weights=weights)
            elif 'wide_resnet50_2' in model_name:
                self.model = torchvision.models.wide_resnet50_2(weights=weights)
            else:
                raise NotImplementedError(
                    'Backbone architecture not recognized!')

        # freeze only if the model is pretrained
        if pretrained:
            if layers_to_freeze >= 0:
                self.model.conv1.requires_grad_(False)
                self.model.bn1.requires_grad_(False)
            if layers_to_freeze >= 1:
                self.model.layer1.requires_grad_(False)
            if layers_to_freeze >= 2:
                self.model.layer2.requires_grad_(False)
            if layers_to_freeze >= 3:
                self.model.layer3.requires_grad_(False)

        # remove the avgpool and most importantly the fc layer
        self.model.avgpool = None
        self.model.fc = None

        if 4 in layers_to_crop:
            self.model.layer4 = None
        if 3 in layers_to_crop:
            self.model.layer3 = None

        out_channels = 2048
        if '34' in model_name or '18' in model_name:
            out_channels = 512
            
        self.out_channels = out_channels // 2 if self.model.layer4 is None else out_channels
        self.out_channels = self.out_channels // 2 if self.model.layer3 is None else self.out_channels

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        if self.model.layer3 is not None:
            x = self.model.layer3(x)
        if self.model.layer4 is not None:
            x = self.model.layer4(x)
        return x



class FeatureMixerLayer(nn.Module):
    def __init__(self, in_dim, mlp_ratio=1):
        super().__init__()
        self.mix = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, int(in_dim * mlp_ratio)),
            nn.ReLU(),
            nn.Linear(int(in_dim * mlp_ratio), in_dim),
        )

        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return x + self.mix(x)



class MixVPR_AGG(nn.Module):
    def __init__(self,
                 in_channels=1024,
                 in_h=20,
                 in_w=20,
                 out_channels=512,
                 mix_depth=1,
                 mlp_ratio=1,
                 out_rows=4,
                 ) -> None:
        super().__init__()

        self.in_h = in_h # height of input feature maps
        self.in_w = in_w # width of input feature maps
        self.in_channels = in_channels # depth of input feature maps
        
        self.out_channels = out_channels # depth wise projection dimension
        self.out_rows = out_rows # row wise projection dimesion

        self.mix_depth = mix_depth # L the number of stacked FeatureMixers
        self.mlp_ratio = mlp_ratio # ratio of the mid projection layer in the mixer block

        hw = in_h*in_w
        self.mix = nn.Sequential(*[
            FeatureMixerLayer(in_dim=hw, mlp_ratio=mlp_ratio)
            for _ in range(self.mix_depth)
        ])
        self.channel_proj = nn.Linear(in_channels, out_channels)
        self.row_proj = nn.Linear(hw, out_rows)

    def forward(self, x):
        x = x.flatten(2)
        x = self.mix(x)
        x = x.permute(0, 2, 1)
        x = self.channel_proj(x)
        x = x.permute(0, 2, 1)
        x = self.row_proj(x)
        x = F.normalize(x.flatten(1), p=2, dim=-1)
        return x


def get_backbone(backbone_arch='resnet50',
                 pretrained=True,
                 layers_to_freeze=2,
                 layers_to_crop=[],):
    """Helper function that returns the backbone given its name

    Args:
        backbone_arch (str, optional): . Defaults to 'resnet50'.
        pretrained (bool, optional): . Defaults to True.
        layers_to_freeze (int, optional): . Defaults to 2.
        layers_to_crop (list, optional): This is mostly used with ResNet where 
                                         we sometimes need to crop the last 
                                         residual block (ex. [4]). Defaults to [].

    Returns:
        nn.Module: the backbone as a nn.Model object
    """
    if 'resnet' in backbone_arch.lower():
        return ResNet(backbone_arch, pretrained, layers_to_freeze, layers_to_crop)

    elif 'efficient' in backbone_arch.lower():
       raise NotImplementedError
            
    elif 'swin' in backbone_arch.lower():
        raise NotImplementedError



def get_aggregator(agg_arch='ConvAP', agg_config={}):
    """Helper function that returns the aggregation layer given its name.
    If you happen to make your own aggregator, you might need to add a call
    to this helper function.

    Args:
        agg_arch (str, optional): the name of the aggregator. Defaults to 'ConvAP'.
        agg_config (dict, optional): this must contain all the arguments needed to instantiate the aggregator class. Defaults to {}.

    Returns:
        nn.Module: the aggregation layer
    """
    
    if 'cosplace' in agg_arch.lower():
        NotImplementedError

    elif 'gem' in agg_arch.lower():
        raise NotImplementedError
    
    elif 'convap' in agg_arch.lower():
        raise NotImplementedError
    
    elif 'mixvpr' in agg_arch.lower():
        assert 'in_channels' in agg_config
        assert 'out_channels' in agg_config
        assert 'in_h' in agg_config
        assert 'in_w' in agg_config
        assert 'mix_depth' in agg_config
        return MixVPR_AGG(**agg_config)




class VPRModel(pl.LightningModule):
    """This is the main model for Visual Place Recognition
    we use Pytorch Lightning for modularity purposes.

    Args:
        pl (_type_): _description_
    """

    def __init__(self,
                #---- Backbone
                backbone_arch='resnet50',
                pretrained=True,
                layers_to_freeze=1,
                layers_to_crop=[],
                
                #---- Aggregator
                agg_arch='ConvAP', #CosPlace, NetVLAD, GeM
                agg_config={},
                
                #---- Train hyperparameters
                lr=0.03, 
                optimizer='sgd',
                weight_decay=1e-3,
                momentum=0.9,
                warmpup_steps=500,
                milestones=[5, 10, 15],
                lr_mult=0.3,
                
                #----- Loss
                loss_name='MultiSimilarityLoss', 
                miner_name='MultiSimilarityMiner', 
                miner_margin=0.1,
                faiss_gpu=False
                 ):
        super().__init__()
        self.encoder_arch = backbone_arch
        self.pretrained = pretrained
        self.layers_to_freeze = layers_to_freeze
        self.layers_to_crop = layers_to_crop

        self.agg_arch = agg_arch
        self.agg_config = agg_config

        self.lr = lr
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.warmpup_steps = warmpup_steps
        self.milestones = milestones
        self.lr_mult = lr_mult

        self.loss_name = loss_name
        self.miner_name = miner_name
        self.miner_margin = miner_margin
        
        self.save_hyperparameters() # write hyperparams into a file
        
        self.batch_acc = [] # we will keep track of the % of trivial pairs/triplets at the loss level 

        self.faiss_gpu = faiss_gpu
        
        # ----------------------------------
        # get the backbone and the aggregator
        self.backbone = get_backbone(backbone_arch, pretrained, layers_to_freeze, layers_to_crop)
        self.aggregator = get_aggregator(agg_arch, agg_config)
        
    # the forward pass of the lightning model
    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregator(x)
        return x


# -------------------------------------------------------------------------------



class MixVPR(BaseTechnique):
    def __init__(self):

        self.name = "mixpvr"

         #weight_pth = package_directory + '/vpr/vpr_techniques/techniques/mixvpr/weights/resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt'
        weight_pth = package_directory + '/weights/resnet50_MixVPR_512_channels(256)_rows(2).ckpt'
        # choose the accelerator
        if torch.cuda.is_available():
            self.device = 'cuda'
            state_dict = torch.load(weight_pth)
        elif torch.backends.mps.is_available():
            self.device = 'cpu'
            state_dict = torch.load(weight_pth, map_location=torch.device('cpu'))
        else:
            self.device = 'cpu'
            state_dict = torch.load(weight_pth)
            

        # Note that images must be resized to 320x320
        self.model = VPRModel(backbone_arch='resnet50',
                        layers_to_crop=[4],
                        agg_arch='mixvpr',
                        agg_config={'in_channels': 1024,
                                    'in_h': 20,
                                    'in_w': 20,
                                    'out_channels': 256,
                                    'mix_depth': 4,
                                    'mlp_ratio': 1,
                                    'out_rows': 2},
                        ).to(self.device)


        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(256, antialias=True),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize(320, antialias=True)
        ])


    def compute_query_desc(self, images: torch.Tensor = None, dataloader: torch.utils.data.dataloader.DataLoader = None, pbar: bool=True) -> dict:
        if images is not None and dataloader is None:
            all_desc = self.model(images.to(self.device)).detach().cpu().numpy()
        elif dataloader is not None and images is None:
            all_desc = []
            for batch in tqdm(dataloader, desc="Computing MixVPR Query Desc", disable=not pbar):
                all_desc.append(self.model(batch.to(self.device)).detach().cpu().numpy())
            all_desc = np.vstack(all_desc)
        else: 
            raise Exception("Can only pass 'images' or 'dataloader'")

        query_desc = {"query_descriptors": all_desc}
        self.set_query(query_desc)
        return query_desc


    def compute_map_desc(self, images: torch.Tensor = None, dataloader: torch.utils.data.dataloader.DataLoader = None, pbar: bool=True) -> dict:
        if images is not None and dataloader is None:
            all_desc = self.model(images.to(self.device)).detach().cpu().numpy()
        elif dataloader is not None and images is None:
            all_desc = []
            for batch in tqdm(dataloader, desc="Computing MixVPR Map Desc", disable=not pbar):
                all_desc.append(self.model(batch.to(self.device)).detach().cpu().numpy())
            all_desc = np.vstack(all_desc)
        else: 
            raise Exception("Can only pass 'images' or 'dataloader'")

        map_desc = {"map_descriptors": all_desc}
        self.set_map(map_desc)
        return map_desc

    def set_map(self, map_descriptors: dict) -> None:
        self.map_desc = map_descriptors
        try: 
            self.map = faiss.IndexFlatIP(map_descriptors["map_descriptors"].shape[1])
            faiss.normalize_L2(map_descriptors["map_descriptors"])
            self.map.add(map_descriptors["map_descriptors"])

        except: 
            self.map = NearestNeighbors(n_neighbors=10, algorithm='auto', 
                                        metric='cosine').fit(map_descriptors["map_descriptors"])

    def set_query(self, query_descriptors: dict) -> None:
        self.query_desc = query_descriptors


    def place_recognise(self, images: torch.Tensor=None, dataloader: torch.utils.data.dataloader.DataLoader = None, top_n: int=1, pbar: bool=True) -> Tuple[np.ndarray, np.ndarray]:
        desc = self.compute_query_desc(images=images, dataloader=dataloader, pbar=pbar)
        if isinstance(self.map, sklearn.neighbors._unsupervised.NearestNeighbors):
            dist, idx = self.map.kneighbors(desc["query_descriptors"])
            return idx[:, :top_n], 1 - dist[:, :top_n]
        else: 
            faiss.normalize_L2(desc["query_descriptors"])
            dist, idx = self.map.search(desc["query_descriptors"], top_n)
            return idx, dist


    def similarity_matrix(self, query_descriptors: dict, map_descriptors: dict) -> np.ndarray:
        return cosine_similarity(map_descriptors["map_descriptors"],
                                 query_descriptors["query_descriptors"]).astype(np.float32)


    def save_descriptors(self, dataset_name: str) -> None:
        if not os.path.isdir(package_directory + "/descriptors/" + dataset_name):
            os.makedirs(package_directory + "/descriptors/" + dataset_name)
        with open(package_directory + "/descriptors/" + dataset_name + "/" + self.name + "_query.pkl", "wb") as f:
            pickle.dump(self.query_desc, f)
        with open(package_directory + "/descriptors/" + dataset_name + "/" + self.name + "_map.pkl", "wb") as f:
            pickle.dump(self.map_desc, f)
        

    def load_descriptors(self, dataset_name: str) -> None:
        if not os.path.isdir(package_directory + "/descriptors/" + dataset_name):
            raise Exception("Descriptor not yet computed for: " + dataset_name)
        with open(package_directory + "/descriptors/" + dataset_name + "/" + self.name + "_query.pkl", "rb") as f:
            self.query_desc = pickle.load(f)
        with open(package_directory + "/descriptors/" + dataset_name + "/" + self.name + "_map.pkl", "rb") as f:
            self.map_desc = pickle.load(f)
            self.set_map(self.map_desc)
        






