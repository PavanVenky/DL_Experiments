import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

#Model Parameters
NUM_EPOCHS = 90
BATCH_SIZE = 128
MOMENTUM = 0.9
LR_DECAY = 0.0005
LR_INIT = 0.01
IMAGE_DIM = 227
NUM_CLASSES = 50
DEVICE_IDS = [0,1,2,3] #GPUS TO USE

CP_DIR = ""

class Alexnet(nn.Module):
    def __init__(self,num_classes):
        super(Alexnet,self).__init__()
        self.feature = nn.Sequential(nn.Conv2d(in_channels=3,out_channels=96,kernel_size=11,stride=4),
                                     nn.ReLU(),
                                     nn.LocalResponseNorm(size=5,alpha=0.0001,beta=0.75,k=2),
                                     nn.MaxPool2d(kernel_size=3,stride=2),

                                     nn.Conv2d(in_channels=96,out_channels=256,kernel_size=5,padding=2),
                                     nn.ReLU(),
                                     nn.LocalResponseNorm(size=5,alpha=0.0001,beta=0.75,k=2),
                                     nn.MaxPool2d(kernel_size=3,stride=2),

                                     nn.Conv2d(in_channels=256,out_channels=384,kernel_size=3,padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=384,out_channels=384,padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=384,out_channels=256,kernel_size=3,padding=1),
                                     nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=3,stride=2))

        self.classifier = nn.Sequential(nn.Dropout(p=0.5,inplace=True),
                                        nn.Linear(in_features=(256*6*6),out_features=4096),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.5,inplace=True),
                                        nn.Linear(in_features=4096,out_features=4096),
                                        nn.ReLU(),
                                        nn.Linear(in_features=4096,out_features=num_classes))

    def forward(self,x):
        out = self.feature(x)
        out = out.view(-1,256*6*6)
        out = self.classifier(out)
        return out


