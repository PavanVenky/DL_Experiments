import torch
import torch.nn as nn

class Lenet(nn.Module):
    def __init__(self,num_classes):
        super(Lenet,self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5,stride=1,padding=0),
                                    nn.BatchNorm2d(num_features=6),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2,stride=2))
        self.layer2 = nn.Sequential(nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5,stride=1,padding=0),
                                    nn.BatchNorm2d(num_features=16),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2,stride=2))
        self.fc1 = nn.Linear(in_features=400,out_features=120)
        self.ac1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=120,out_features=84)
        self.ac2 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=84,out_features=num_classes)
        self.ac3 = nn.ReLU()

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)
        out = self.fc1(out)
        out = self.ac1(out)
        out = self.fc2(out)
        out = self.ac2(out)
        out = self.fc3(out)
        out = self.ac3(out)
        return out
    

class Alexnet(nn.Module):
    def __init__(self,num_classes):
        super(Alexnet,self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(in_channels=3,out_channels=96,kernel_size=11,stride=4,padding=0),
                                    nn.BatchNorm2d(num_features=96),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=3,stride=2))
        self.layer2 = nn.Sequential(nn.Conv2d(in_channels=96,out_channels=256,kernel_size=5,stride=1,padding=2),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=3,stride=2))
        self.layer3 = nn.Sequential(nn.Conv2d(in_channels=256,out_channels=384,kernel_size=3,padding=1,stride=1),
                                    nn.BatchNorm2d(384),
                                    nn.ReLU())
        self.layer4 = nn.Sequential(nn.Conv2d(in_channels=384,out_channels=384,kernel_size=3,stride=1,padding=1),
                                    nn.BatchNorm2d(384),
                                    nn.ReLU())
        self.layer5 = nn.Sequential(nn.Conv2d(in_channels=384,out_channels=256,kernel_size=3,stride=1,padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=3,stride=2),
                                    nn.Dropout(0.5))
        self.fc1    = nn.Sequential(nn.Dropout(0.5),
                                    nn.Linear(in_features=6400,out_features=4096),
                                    nn.ReLU())
        self.fc2    = nn.Sequential(nn.Dropout(0.5),
                                    nn.Linear(4096,4096),
                                    nn.ReLU())
        self.fc3    = nn.Sequential(nn.Linear(4096,num_classes))

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.view(out.size(0),-1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)

        return out
