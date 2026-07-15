import os
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from PIL import Image


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

    # plt.figure(figsize=(10, 6))
    # bars = plt.bar(class_ids, image_counts, color="skyblue", edgecolor="black")
    # plt.xlabel("Class ID")
    # plt.ylabel("Image Count")
    # plt.title("Images per Class")
    # plt.xticks(class_ids, class_ids)
    # plt.grid(axis="y", alpha=0.3)

    # max_count = max(image_counts) if image_counts else 0
    # offset = max(1, max_count * 0.01)
    # for bar, count, text in zip(bars, image_counts, label_texts):
    #     height = bar.get_height()
    #     plt.text(
    #         bar.get_x() + bar.get_width() / 2,
    #         height + offset,
    #         str(count),
    #         ha="center",
    #         va="bottom",
    #         fontsize=10,
    #         fontweight="bold",
    #         rotation=0,
    #     )
    #     plt.text(
    #         bar.get_x() + bar.get_width() / 2,
    #         height + offset * 4,
    #         text,
    #         ha="center",
    #         va="bottom",
    #         fontsize=9,
    #         rotation=0,
    #     )

    # plt.tight_layout()
    # plt.show()

    image_paths = []
    image_labels = []
    for classid in class_ids:
        label = labels.get(classid, f"class {classid}")
        print(f"class {classid} ({label}): {counts[classid]['count']} images")
        for path in counts[classid]["paths"]:
            image_paths.append(path)
            image_labels.append(classid)

    return counts, image_paths, image_labels



#get images and labels
path = r"D:\Pavan\My_Python_Project\DL_Experiments\DL_Experiments\Datasets\Traffic_Sign_Dataset\traffic_Data\DATA"
fulldata,imagepaths,imagelabels = count_images_by_class(path)

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
        img = Image.open(self.imagesFolder[index]).convert("RGB")
        label = torch.tensor(self.imagesLabels[index],dtype=torch.long)

        if self.transform:
            img = self.transform(img)

        return img,label
    
train_data = Mydata(imagepaths,imagelabels,transform)
train_data_loader = DataLoader(train_data,batch_size=32,shuffle=True,num_workers=4)

for images,lables in train_data_loader:
    print(images.shape)
    print(lables.shape)