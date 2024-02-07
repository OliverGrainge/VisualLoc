import numpy as np
from PlaceRec.Training.models import aggregators
from PlaceRec.Training.models import backbones
import torchvision
import torch

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
        return backbones.ResNet(backbone_arch, pretrained, layers_to_freeze, layers_to_crop)

    elif 'efficient' in backbone_arch.lower():
        if '_b' in backbone_arch.lower():
            return backbones.EfficientNet(backbone_arch, pretrained, layers_to_freeze+2)
        else:
            return backbones.EfficientNet(model_name='efficientnet_b0',
                                          pretrained=pretrained, 
                                          layers_to_freeze=layers_to_freeze)
    elif 'mobilenet' in backbone_arch.lower():
        return backbones.MobileNetV2(backbone_arch, pretrained, layers_to_freeze)
    elif 'squeezenet' in backbone_arch.lower():
        return backbones.SqueezeNet(backbone_arch, pretrained, layers_to_freeze)
    elif 'vgg16' in backbone_arch.lower():
        return backbones.VGG16(backbone_arch, pretrained, layers_to_freeze)
    elif "dinov2" in backbone_arch.lower():
        vit = backbones.dinov2_vitb14(pretrained=True)
        vit.init_pos_encoding()
        for param in vit.parameters(): 
            param.requires_grad = False
        vit = vit.cuda()
        return vit

def get_aggregator(agg_arch, feature_map_shape, out_dim=1024):
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
        return aggregators.CosPlace(feature_map_shape, out_dim)

    elif 'gem' in agg_arch.lower():
        return aggregators.GeMPool(feature_map_shape, out_dim)
    
    elif 'convap' in agg_arch.lower():
        return aggregators.ConvAP(feature_map_shape, out_dim)
    
    elif 'mixvpr' in agg_arch.lower():
        return aggregators.MixVPR(feature_map_shape, out_dim)
    
    elif 'spoc' in agg_arch.lower(): 
        return aggregators.SPoC(feature_map_shape, out_dim)
    
    elif 'mac' in agg_arch.lower():
        return aggregators.MAC(feature_map_shape, out_dim)
    
    elif 'netvlad' in agg_arch.lower(): 
        return aggregators.NetVLAD(feature_map_shape, out_dim)


def get_model(backbone_arch, agg_arch, descriptor_size=1024, pretrained=True):
    backbone = get_backbone(backbone_arch, pretrained)
    img = torch.randn(1, 3, 320, 320)
    feature_map_shape = backbone(img)[0].shape
    aggregator = get_aggregator(agg_arch, feature_map_shape, out_dim=descriptor_size)
    return torch.nn.Sequential(backbone, aggregator)
