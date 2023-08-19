import torch.nn as nn
import torch.nn.init as init
import torch


class HybridNet(nn.Module):
    def __init__(self):
        super(HybridNet, self).__init__()

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
        self.fc7_new = nn.Linear(256 * 6 * 6, 4096)  # Assuming the spatial size is 6x6 after the pooling layers
        init.normal_(self.fc7_new.weight, std=0.005)
        init.constant_(self.fc7_new.bias, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout(p=0.5)

        # FC8
        self.fc8_new = nn.Linear(4096, 2543)
        init.normal_(self.fc8_new.weight, std=0.01)
        init.constant_(self.fc8_new.bias, 0)
        self.prob = nn.Softmax(dim=1)

    def forward(self, x):
        # Define the forward pass based on the layers and activations
        x = self.norm1(self.pool1(self.relu1(self.conv1(x))))
        x = self.norm2(self.pool2(self.relu2(self.conv2(x))))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.conv5(x)

        #implement spatial pooling
        return x

# Create the model
model = HybridNet()

my_state_dict = model.state_dict()
weights = "/home/oliver/Documents/github/VisualLoc/PlaceRec/Methods/weights/AmosNet.caffemodel.pt"

state_dict = torch.load(weights)

model.load_state_dict(state_dict)

"""
model.load_state_dict(state_dict)

layers = list(state_dict.keys())
my_layers = list(my_state_dict.keys())

for i in range(len(layers)):
    print(layers[i], my_layers[i])

print("===================================")


for i in range(len(layers)):
    print(state_dict[layers[i]].shape, my_state_dict[my_layers[i]].shape)



#model = HybridNet()

#x = torch.rand(10, 3, 227, 227)

#model(x)



"""