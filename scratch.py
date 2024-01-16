import torchvision.models as models
from PlaceRec.Methods import ResNet50ConvAP
import torch
import pytorch_lightning as pl

method = ResNet50ConvAP(pretrained=False)
model = method.model.cpu()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import numpy as np
import torchvision
from torchvision import transforms 

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
        att = self.softmax(att)

        features = F.relu(self.reduce_conv(fused_features))

        weighted_features = features * att.unsqueeze(-1).unsqueeze(-1)
        features = weighted_features + x
        features = F.relu(self.bn(features))
        return features




class GatingResNetModel(torch.nn.Module):
    def __init__(self, out_dim=100):
        super(GatingResNetModel, self).__init__()
        # Copy layers from original model up to the truncation point
        self.conv1 = nn.Conv2d(3, 256, 3, 2)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 512, 3, 1)
        self.bn2 = nn.BatchNorm2d(512)
        self.gatinginception = InceptionBlock(512, 512)
        self.AAP = nn.AdaptiveAvgPool2d(2)
        self.fc = nn.Linear(512*4, out_dim)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        b, c, h, w = x.shape
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.gatinginception(x)
        x = self.AAP(x).view(b, -1)
        x = self.fc(x)
        x = self.softmax(x)
        return x
    

    
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)



trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform)
testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                       download=True, transform=transform)

class UniformRandomResizeDataset(data.Dataset):
    def __init__(self, basedistillationdataset, mult_range=(0.8, 1.0), teacher_size=(32, 32)):
        self.base_dataset = basedistillationdataset
        self.teacher_size = teacher_size
        self.mult_range = mult_range

    def __len__(self):
        return self.base_dataset.__len__()

    def __getitem__(self, idx):
        img, target = self.base_dataset.__getitem__(idx)
        return img, target

    def collate_fn(self, batch):
        random_mult = self.mult_range[0] + (np.random.rand() * (self.mult_range[1] - self.mult_range[0]))
        res = (int(random_mult * self.teacher_size[0]), int(random_mult * self.teacher_size[1]))
        resize_transform = transforms.Resize(res, antialias=True)
        imgs = [resize_transform(item[0]) for item in batch]
        targets = [torch.tensor(item[1]) for item in batch]
        imgs = torch.stack(imgs, dim=0)
        targets = torch.vstack(targets)
        return imgs, targets
    



class ModelModule(pl.LightningModule):
    def __init__(self, model):
        super(ModelModule, self).__init__()
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        imgs, target_features = batch
        features = self.model(imgs)
        loss = self.loss_fn(features, target_features.squeeze())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, target_features = batch
        features = self.model(imgs)
        loss = self.loss_fn(features, target_features.squeeze())
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch):
        imgs, target_features = batch
        features = self.model(imgs)
        loss = self.loss_fn(features, target_features.squeeze())
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        return optimizer
    

trainset = UniformRandomResizeDataset(trainset)
testset = UniformRandomResizeDataset(testset)
trainloader = data.DataLoader(trainset, batch_size=32, collate_fn=trainset.collate_fn)
valloader = data.DataLoader(testset, batch_size=32, collate_fn=testset.collate_fn)

model = GatingResNetModel(100)
model = ModelModule(model)

trainer = pl.Trainer(
    max_epochs=100,
    accelerator="gpu"
)

trainer.fit(model, trainloader, valloader)