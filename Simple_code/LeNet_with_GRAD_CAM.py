import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

from keras.layers import Conv2D,MaxPooling2D,Flatten,Input
from keras.models import Sequential,Model
from keras.losses import CategoricalCrossentropy
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report

print(cv2.__version__)

path = "Datasets\Cricket_Legends_data"

folders_path = os.path.join(path,os.listdir(path)[0])

print(folders_path)
classes = os.listdir(folders_path)

labels = []

# LABELS DISPLAY AND ANALYSIS
# for player in classes:
#     player_dir = os.path.join(folders_path,player)
#     player_img_cnt = len(os.listdir(player_dir))
#     labels.extend([player]*player_img_cnt)

# classes, counts = np.unique(labels, return_counts=True)
# plt.figure(figsize=(10, 6))
# bars = plt.barh(classes, counts, color='skyblue')
# plt.xlabel('Total image count')
# plt.ylabel('Class name')
# plt.title('Images per player class')

# for bar, count in zip(bars, counts):
#     plt.text(
#         bar.get_width() + 1,      # x position just after the bar
#         bar.get_y() + bar.get_height() / 2,
#         str(count),
#         va='center',
#         ha='left'
#     )

# plt.tight_layout()
# plt.show()

#PRE-PROCESS AND DATA SPLIT
X = []
Y = []
label_name = {}

for idx,player in enumerate(classes):
    player_dir = os.path.join(folders_path,player)
    player_img_cnt = len(os.listdir(player_dir))
    label_name[player] = idx
    for img_name in os.listdir(player_dir):
        each_player_img_path = os.path.join(player_dir,img_name)
        img = cv2.imread(each_player_img_path)
        img = cv2.resize(img,(128,128))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        X.append(img)
        Y.append(idx)

#https://www.kaggle.com/code/evilspirit05/lenet-cricketer-classifier-grad-cam-maximization#summary-of-our-data

#convert lists to numpy
X = np.array(X)
Y = np.array(Y)

X = X/255.0 #normalize data

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=42)
print(f"Total Images: {len(X)} Train data = {len(x_train)} test data = {len(x_test)}")
print(f"labels: {label_name}")
