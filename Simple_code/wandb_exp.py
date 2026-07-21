import torch
import torch.nn as nn
import torchvision
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import wandb



# epochs = 200
# batchsize = 32
# lr = 0.001

class Mylenet(nn.Module):
    def __init__(self):
        super(Mylenet,self).__init__()
        self.backbone = nn.Sequential(nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5,stride=1),
                                      nn.Tanh(),
                                      nn.AvgPool2d(kernel_size=2,stride=2),

                                      nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5,stride=1),
                                      nn.Tanh(),
                                      nn.AvgPool2d(kernel_size=2,stride=2))
        self.fclayer = nn.Sequential(nn.Linear(in_features=256,out_features=120),
                                     nn.Tanh(),
                                     nn.Linear(in_features=120,out_features=84),
                                     nn.Tanh(),
                                     nn.Linear(in_features=84,out_features=10))
        
    def forward(self,x):
        out = self.backbone(x)
        out = out.view(-1, 16 * 4 * 4)
        out = self.fclayer(out)
        return out


def train(model,train_dataloader,device,cost_fn,optimizer):
    running_loss = 0.0
    total = 0.0
    correct = 0.0

    model.train()
    for idx,(image,label) in enumerate(train_dataloader):
        image = image.to(device)
        label = label.to(device)

        output = model(image)
        loss = cost_fn(output,label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += label.size(0)
        correct += predicted.eq(label).sum().item()

    train_loss = running_loss / len(train_dataloader)
    train_acc = 100. * correct / total
    return train_loss, train_acc

def test(model,test_dataloader,device,cost_fn):
    total_loss = 0.0
    total = 0.0
    correct = 0.0

    model.eval()
    with torch.no_grad():
        for idx,(image,label) in enumerate(test_dataloader):
            image,label = image.to(device), label.to(device)
            output = model(image)
            loss = cost_fn(output,label)
            total_loss += loss.item()
            _,predicted = output.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()

    test_loss = total_loss / len(test_dataloader)
    test_acc = 100. * correct / total
    return test_loss, test_acc

def main():
    config = {"epochs":200,
              "batch_size":32,
              "lr":0.001,
              "architecture":"Lenet",
              "dataset":"MNIST"}
    wandb.config = config

    wandb.init(project="lenet_experiment",config=config)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # MNIST mean and std
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = datasets.MNIST(root="./data",train=True,transform=transform,download=True)
    test_data = datasets.MNIST(root="./data",train=False,transform=transform,download=True)

    train_dataloader = DataLoader(train_data,batch_size=config["batch_size"],shuffle=True)
    test_dataloader = DataLoader(test_data,batch_size=config["batch_size"],shuffle=True)

    model = Mylenet().to(device)
    cost_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),lr=config["lr"],momentum=0.9)
    wandb.watch(model, cost_fn, log="all", log_freq=100)

    for epoch in range(config["epochs"]):
        train_loss,train_accuracy = train(model,train_dataloader,device,cost_fn,optimizer)
        test_loss,test_accuracy = test(model,test_dataloader,device,cost_fn)
        print(f"Epoch {epoch+1}/{config['epochs']} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_accuracy:.2f}%")
        
        wandb.log({
            "epoch":epoch,
            "train_loss":train_loss,
            "train_acc":train_accuracy,
            "test_loss":test_loss,
            "test_acc":test_accuracy
        })

if __name__ == "__main__":
    main()