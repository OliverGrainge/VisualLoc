import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl




#conv2d(in_channels, out_channels, kernel_size, stride=, padding=)
class HybridNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4)
        self.pool1 = nn.MaxPool2d(kernel_size=(3,3), stride=2)
        self.LRN1 = nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75)

        self.conv2 = nn.Conv2d(96, 256, 5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=(3,3), stride=2)
        self.LRN2 = nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75)

        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=0)

        self.conv4 = nn.Conv2d(384, 384, 3)

        self.conv5 = nn.Conv2d(384, 256, 3)

        self.conv6 = nn.Conv2d(256, 256, 3)
        self.pool6 = nn.MaxPool2d(kernel_size=(3,3), stride=2)


        self.fc7_new = nn.Linear(9216, 4096)
        self.drop7 = nn.Dropout(p=0.5)

        self.fc8_new = nn.Linear(4096, 2543)
        self.softmax = nn.Softmax(dim=1)
        



    def forward(self, x):
        x = F.relu(self.conv1(x))
        print('==================', x.shape)
        x = self.pool1(x)
        x = self.LRN1(x)
        
        x = F.relu(self.conv2(x))
        print('==================', x.shape)
        x = self.pool2(x)
        x = self.LRN2(x)

        x = F.relu(self.conv3(x))
        print('=======================', x.shape)
        x = F.relu(self.conv4(x))
        print('==================', x.shape)

        x = F.relu(self.conv5(x))
        print('==================', x.shape)

        x = F.relu(self.conv6(x))
        print('==================', x.shape)
        x = self.pool6(x)
        x = x.flatten(start_dim=1)
        print('==================', x.shape)

        x = F.relu(self.fc7_new(x))
        x = self.drop7(x)
        x = self.fc8_new(x)
        
        x = self.softmax(x)
        return x

weights = "/home/oliver/Documents/github/VisualLoc/PlaceRec/Methods/weights/AmosNet.caffemodel.pt"

state_dict = torch.load(weights)

for name, weights in state_dict.items():
    print (name, weights.numpy().shape)

print("===================================")
#model = HybridNet()

#x = torch.rand(10, 3, 227, 227)

#model(x)




