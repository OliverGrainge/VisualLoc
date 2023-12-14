from transformers import ViTModel, ViTFeatureExtractor
from PIL import Image
import requests
import torch.nn as nn
from torchvision import transforms



preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224
    transforms.ToTensor(),  # This will automatically scale pixels to [0, 1]
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])

class vit_base_patch16_224_gap(nn.Module):
    def __init__(self, feature_dim=2048, num_trainable_blocks=3):
        super().__init__()
        self.vit_backbone = ViTModel.from_pretrained('google/vit-base-patch16-224')

        # Freeze all transformer blocks except the last `num_trainable_blocks`
        num_blocks = len(self.vit_backbone.encoder.layer) # Total number of blocks
        for i, block in enumerate(self.vit_backbone.encoder.layer):
            if i < num_blocks - num_trainable_blocks:
                for param in block.parameters():
                    param.requires_grad = False


        self.fc = nn.Linear(768, feature_dim)
    
    def forward(self, x):
        x = self.vit_backbone(x).last_hidden_state
        x = x.mean(1)
        x = self.fc(x)
        return x


url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)
image = preprocess(image)
model = vit_base_patch16_224_gap()
features = model(image[None, :])

print(features.shape)