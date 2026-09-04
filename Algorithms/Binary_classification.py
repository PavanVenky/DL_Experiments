import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms,datasets

IMAGE_SIZE=64
EPOCHS=300
LR=0.001
BATCH_SIZE=32

device = 'cpu'
if torch.cuda.is_available():
    device='cuda'

transform = transforms.Compose([transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
                                transforms.ToTensor()])

train_data = datasets.ImageFolder(root="dataset/train",transform=transform)
test_data = datasets.ImageFolder(root="dataset/test",transform=transform)

print("classes :",train_data.classes)
print("class mappint: ",train_data.class_to_idx)

train_loader = DataLoader(train_data,shuffle=True,batch_size=BATCH_SIZE)
test_loader = DataLoader(test_data,shuffle=True,batch_size=BATCH_SIZE)


class image_cassifier(nn.Module):
    def __init__(self):
        super(image_cassifier).__init__()
        self.network = nn.Sequential(nn.Linear(in_features=64*64*3,out_features=128),
                                     nn.ReLU(),

                                     nn.Linear(in_features=128,out_features=64),
                                     nn.ReLU(),

                                     nn.Linear(out_features=64,in_features=32),
                                     nn.ReLU(),

                                     nn.Linear(in_features=64,out_features=1))

    def forward(self,x):
        x = x.view(x.size(0), -1) # Input: [Batch, 3, 64, 64] output: [batch,12288]
        x = self.network(x) # input: [batch,12288] output: [batch,1]
        return x


model = image_cassifier().to(device=device)
print(model)

loss_fn = nn.BCEWithLogitsLoss()
criterion = torch.optim.Adam(model.parameters(),lr=LR)

model.train()

for epoch in range(EPOCHS):
    running_loss=0.0
    correct=0
    total=0

    for image,label in enumerate(train_loader):
        image=image.to(device)
        label=label.float().to(device)

        output = model(image) # outputs shape: # [Batch, 1]
        output = output.squeeze(1) # outputs shape: [Batch]

        loss = loss_fn(output,label)

        criterion.zero_grad()
        loss_fn.backward()
        criterion.step()

        probabilities = torch.sigmoid(output)
        predictions=(probabilities>=0.5).float()
        correct+=(predictions==label).sum().item()
        total+=label.size(0)
        running_loss+=loss.item()

    accuracy = correct/total
    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {running_loss / len(train_loader):.4f} "
        f"Accuracy: {accuracy:.4f}")

