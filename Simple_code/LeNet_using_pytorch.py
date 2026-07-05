import torch
import torch.nn as nn
from torchvision.transforms import transforms
import torchvision

#https://www.digitalocean.com/community/tutorials/writing-lenet5-from-scratch-in-python#testing-accuracy

epochs = 10
batchsize = 64
learning_rate = 0.001
classes_cnt = 10

device = 'cpu'
if torch.cuda.is_available():
    device = 'gpu'


class Lenet_network(nn.Module):
    def __init__(self,num_classes):
        super(Lenet_network,self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5,stride=1,padding=0),
                                    nn.BatchNorm2d(6),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2,stride=2))
        self.layer2 = nn.Sequential(nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5,stride=1,padding=0),
                                    nn.BatchNorm2d(16),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2,stride=2))
        self.fc1    = nn.Linear(in_features=400,out_features=120)
        self.ac1    = nn.ReLU()
        self.fc2    = nn.Linear(in_features=120,out_features=84)
        self.ac2    = nn.ReLU()
        self.fc3    = nn.Linear(in_features=84,out_features=num_classes)
        self.ac3    = nn.ReLU()

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)  # Flatten
        out = self.fc1(out)
        out = self.ac1(out)
        out = self.fc2(out)
        out = self.ac2(out)
        out = self.fc3(out)
        out = self.ac3(out)
        return out

transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.MNIST(
    root = "./data",
    train=False,
    download=True,
    transform=transform
)

train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,shuffle=True,batch_size=batchsize)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,shuffle=True,batch_size=batchsize)

model = Lenet_network(classes_cnt).to(device)

cost = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(),lr=learning_rate)

total_step = len(test_dataloader)

for epoch in range(epochs):
    for i,(images,lables) in enumerate(train_dataloader):
        #fwd pass
        images = images.to(device)
        lables = lables.to(device)
        outputs = model(images)
        loss = cost(outputs,lables)

        #backward pass
        optim.zero_grad()
        loss.backward()
        optim.step()
        if (i+1) % 400 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, i+1, total_step, loss.item()))


model.eval() #set model in eval mode

with torch.no_grad():
    correct = 0
    total = 0
    
    for images,labels in test_dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _,predicted = torch.max(outputs,1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy of the network on 10000 test images: {accuracy:.2f} %")