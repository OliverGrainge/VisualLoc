import torch
from torchvision import transforms



preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((480, 640), antialias=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class NetVLAD(BaseModelWrapper):
    def __init__(self, pretrained: bool = True):
        model = model = torch.hub.load("gmberton/eigenplaces", "get_trained_model", backbone="ResNet50", fc_output_dim=2048)
        if pretrained:
         

        super().__init__(model=model, preprocess=preprocess, name="netvlad")
