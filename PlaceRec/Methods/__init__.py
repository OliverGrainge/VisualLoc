from .amosnet import AmosNet
from .anyloc import AnyLoc
from .base_method import SingleStageBaseModelWrapper, TwoStageBaseModelWrapper
from .cct_cls import CCT_CLS
from .cct_netvlad import CCT_NetVLAD
from .cct_seqpool import CCT_SEQPOOL
from .convap import (
    ConvAP,
    ResNet34_ConvAP,
    ResNet18_ConvAP,
    MobileNetV2_ConvAP,
    ShuffleNetV2_ConvAP,
    SqueezeNetV1_ConvAP,
)
from .cosplace import CosPlace
from .dino_salad import DinoSalad
from .dinov2b14_cls import DINOv2B14_CLS
from .dinov2s14_cls import DINOv2S14_CLS
from .eigenplaces import EigenPlaces
from .hybridnet import HybridNet
from .mixvpr import MixVPR, ResNet34_MixVPR, ResNet18_MixVPR, MobileNetV2_MixVPR
from .netvlad import NetVLAD, ResNet34_NetVLAD, ResNet18_NetVLAD, MobileNetV2_NetVLAD
from .gem import ResNet50_GeM, ResNet34_GeM, ResNet18_GeM, MobileNetV2_GeM
from .selavpr import SelaVPR
from .sfrs import SFRS
from .vit_cls import ViT_CLS
from .vit_salad import ViTSalad
