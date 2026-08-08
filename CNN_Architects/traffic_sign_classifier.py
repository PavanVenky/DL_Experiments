import os
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
import cv2
from PIL import Image
from All_CNN_arcitects import Alexnet

try:
    import wandb
except ImportError:
    wandb = None

def load_labels(labels_csv_path):
    labels = {}
    if not os.path.isfile(labels_csv_path):
        return labels

    with open(labels_csv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [part.strip() for part in line.split(",", 1)]
            if len(parts) < 2:
                continue
            try:
                classid = int(parts[0])
            except ValueError:
                continue
            labels[classid] = parts[1]
    return labels


def count_images_by_class(folder_path):
    """
    Count images per class in a folder where filenames are like:
    00x_imagename.jpg, with x = class id (0..7).
    Display a bar chart with matplotlib and annotate each bar with the class label.
    """
    counts = {i: {"count": 0, "paths": []} for i in range(8)}

    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if not os.path.isfile(filepath):
            continue

        if len(filename) >= 4 and filename[3] == "_":
            try:
                classid = int(filename[2])
            except ValueError:
                continue
            if 0 <= classid <= 7:
                counts[classid]["count"] += 1
                counts[classid]["paths"].append(filepath)

    labels_csv = os.path.join(os.path.dirname(os.path.dirname(folder_path)), "labels.csv")
    labels = load_labels(labels_csv)

    class_ids = sorted(counts.keys())
    image_counts = [counts[classid]["count"] for classid in class_ids]
    label_texts = [labels.get(classid, f"class {classid}") for classid in class_ids]

    image_paths = []
    image_labels = []
    for classid in class_ids:
        label = labels.get(classid, f"class {classid}")
        print(f"class {classid} ({label}): {counts[classid]['count']} images")
        for path in counts[classid]["paths"]:
            image_paths.append(path)
            image_labels.append(classid)
    print("*****************************")

    return counts, image_paths, image_labels



#get images and labels
path = r"D:\Pavan\My_Python_Project\DL_Experiments\DL_Experiments\Datasets\Traffic_Sign_Dataset\traffic_Data\DATA"
test_path = r"D:\Pavan\My_Python_Project\DL_Experiments\DL_Experiments\Datasets\Traffic_Sign_Dataset\traffic_Data\TEST"

fulldata,imagepaths,imagelabels = count_images_by_class(path)
testdata,imagepaths_test,imagelabels_test = count_images_by_class(test_path)

transform = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor()])

class Mydata(Dataset):
    def __init__(self,images_data,images_labels,transform=None):
        self.imagesFolder = images_data
        self.imagesLabels = images_labels
        self.transform = transform

    def __len__(self):
        return len(self.imagesFolder)
    
    def __getitem__(self, index):
        img = cv2.imread(self.imagesFolder[index])
        if img is None:
            raise ValueError(f"Could not read image: {self.imagesFolder[index]}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        label = torch.tensor(self.imagesLabels[index], dtype=torch.long)

        if self.transform:
            img = self.transform(img)

        return img, label
    
train_data = Mydata(imagepaths,imagelabels,transform)
train_data_loader = DataLoader(train_data,batch_size=32,shuffle=True)

test_data = Mydata(imagepaths_test,imagelabels_test,transform)
test_data_loader = DataLoader(test_data,batch_size=32,shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# for images,lables in train_data_loader:
#     print(images.shape)
#     print(lables.shape)

# print("********")

# for images,lables in test_data_loader:
#     print(images.shape)
#     print(lables.shape)

num_classes = len(set(imagelabels))
model = Alexnet(num_classes=num_classes).to(device)

optim = torch.optim.AdamW(model.parameters(),lr=0.001)
cost_fn = nn.CrossEntropyLoss()

# training start
epochs = 200
batch_size = 32
learning_rate = 0.001
total_step = len(train_data_loader)

config = {
    "epochs": epochs,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "architecture": "Alexnet",
    "dataset": "Traffic Sign Dataset"
}

if wandb is not None:
    wandb.init(project="traffic_sign_classifier", config=config)
    wandb.watch(model, criterion=cost_fn, log="all", log_freq=100)

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0

    for i, (images, labels) in enumerate(train_data_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = cost_fn(outputs, labels)

        optim.zero_grad()
        loss.backward()
        optim.step()

        epoch_loss += loss.item()
        print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, i+1, total_step, loss.item()))

    avg_epoch_loss = epoch_loss / total_step
    print(f"Epoch [{epoch+1}/{epochs}] Average Loss: {avg_epoch_loss:.4f}")

    if wandb is not None:
        wandb.log({"epoch": epoch + 1, "train_loss": avg_epoch_loss})


#Test pipeline

model.eval()

correct = 0.0
total = 0.0
for i,(images,lables) in enumerate(test_data_loader):
    images = images.to(device)
    labels = lables.to(device)
    test_output = model(images)
    probabilities = torch.softmax(test_output,dim=1)
    _, predicted = torch.max(probabilities, dim=1)
    
    correct += (predicted==labels).sum().item()
    total += lables.size(0)

accuracy = 100.0 * correct/total
print(f"Accuracy of the network on {total} test images: {accuracy:.2f}%")
