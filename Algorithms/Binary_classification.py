import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms,datasets

IMAGE_SIZE=128
EPOCHS=300
LR=0.001
BATCH_SIZE=32

device = 'cpu'
if torch.cuda.is_available():
    device='cuda'

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder(root=r"Datasets\PetImages\train", transform=transform)
test_data = datasets.ImageFolder(root=r"Datasets\PetImages\test", transform=transform)

print("classes :",train_data.classes)
print("class mappint: ",train_data.class_to_idx)

train_loader = DataLoader(train_data,shuffle=True,batch_size=BATCH_SIZE)
test_loader = DataLoader(test_data, shuffle=False, batch_size=BATCH_SIZE)


class image_cassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn     = nn.Sequential(nn.Conv2d(in_channels=3,kernel_size=5,out_channels=6,stride=1),
                                     nn.BatchNorm2d(num_features=6),
                                     nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=2,stride=2),

                                     nn.Conv2d(in_channels=6,kernel_size=5,stride=1,out_channels=16),
                                     nn.BatchNorm2d(num_features=16),
                                     nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=2,stride=2))

        self.network = nn.Sequential(nn.Linear(in_features=16*29*29,out_features=128),
                                     nn.ReLU(),

                                     nn.Linear(in_features=128,out_features=64),
                                     nn.ReLU(),

                                     nn.Linear(in_features=64,out_features=32),
                                     nn.ReLU(),

                                     nn.Linear(in_features=32,out_features=1))

    def forward(self,x):
        x = self.cnn(x)
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

    for image,label in train_loader:
        image = image.to(device)
        label = label.float().to(device)

        output = model(image) # outputs shape: # [Batch, 1]
        output = output.squeeze(1) # outputs shape: [Batch]

        loss = loss_fn(output,label)

        criterion.zero_grad()
        loss.backward()
        criterion.step()

        probabilities = torch.sigmoid(output)
        predictions=(probabilities>=0.5).float()
        correct+=(predictions==label).sum().item()
        total+=label.size(0)
        running_loss+=loss.item()
        print(f"Correct: {(predictions==label).sum().item()} out of Batch: {BATCH_SIZE}")

    accuracy = correct/total
    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {running_loss / len(train_loader):.4f} "
        f"Accuracy: {accuracy:.4f}")