import torch
import torch.nn as nn
from init import weights_init_kaiming_normal

class LeNet5(nn.Module):
    # define LetNet5
    def __init__(self, num_classes):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5,stride=1,padding=2)
        self.acti1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        self.conv2 = nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5,stride=1,padding=0)
        self.acti2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        self.conv2_1 = nn.Conv2d(in_channels=16,out_channels =16, kernel_size=3, stride = 1, padding =1)
        self.acti2_1 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        self.fc3 = nn.Linear(in_features=3600,out_features=120)
        self.acti3 = nn.ReLU()
        self.fc4 = nn.Linear(in_features=120,out_features=90)
        self.acti4 = nn.ReLU()
        self.fc5 = nn.Linear(in_features=90,out_features=84)
        self.acti5 = nn.ReLU()
        self.fc6 = nn.Linear(in_features=84,out_features=62)
        weights_init_kaiming_normal(self)

    def forward(self, x):
        # the first convolutional, activation and maxpooling layer
        x = self.conv1(x)
        x = self.acti1(x)
        x = self.maxpool1(x)

        # the second convolutional, activation and maxpooling layer
        x = self.conv2(x)
        x = self.acti2(x)
        x = self.maxpool2(x)

        x = self.conv2_1(x)
        x = self.acti2_1(x)
        x = self.maxpool3(x)
        # stack the activation maps into 1d vector
        x = x.view(-1, 3600)

        # third fully-connected (fc) layer and activation layer
        x = self.fc3(x)
        x = self.acti3(x)

        # fourth fully-connected layer and activation layer
        x = self.fc4(x)
        x = self.acti4(x)

        # last fc layer
        x = self.fc5(x)
        x = self.acti5(x)
        y = self.fc6(x)

        return y