import cv2
import os
import numpy as np
import selectivesearch as ss
import matplotlib.pyplot as plt

def get_region_proposals(img):
    img_lbl, proposals = ss.selective_search(img, min_size=1000)

    bounding_boxes = [region['rect'] for region in proposals]

    final_region_proposals = []
    for box in bounding_boxes:
        x,y,w,h = box
        final_region_proposals.append([x,y,x+w,y+h])

    return final_region_proposals

def calculate_iou(bb_1, bb_2):
    # bb_1 and bb_2 are [x1, y1, x2, y2]
    x_left = max(bb_1[0], bb_2[0])
    y_top = max(bb_1[1], bb_2[1])
    x_right = min(bb_1[2], bb_2[2])
    y_bottom = min(bb_1[3], bb_2[3])
    
    if x_right > x_left and y_bottom > y_top:
        intersection = (x_right - x_left) * (y_bottom - y_top)
    else:
        intersection = 0
    
    area1 = (bb_1[2] - bb_1[0]) * (bb_1[3] - bb_1[1])
    area2 = (bb_2[2] - bb_2[0]) * (bb_2[3] - bb_2[1])
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0
    return intersection / union

def get_best_BB_list(region_proposals,gnd_truth):
    positiveSamples = []
    negativeSamples = []
    training_labels = []
    for bb in region_proposals:
        iou_value = calculate_iou(bb,gnd_truth)
        if iou_value > 0.4:
            positiveSamples.append(bb)
            training_labels.append(1)
        else:
            negativeSamples.append(bb)
            training_labels.append(0)

    return positiveSamples,negativeSamples,training_labels


def check_annotations(img_path,txt_path):
    img = cv2.imread(img_path)
    img_h,img_w = img.shape[:2]

    with open(txt_path,'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts)!=5:
            print("No data ..")
            continue
        try:
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            # YOLO Format
            tlx = int((x_center - width/2)*img_w)
            tly = int((y_center - height/2)*img_h)
            brx = int((x_center + width/2)*img_w)
            bry = int((y_center + height/2)*img_h)

            cv2.rectangle(img,(tlx,tly),(brx,bry),(0,255,0),2)
        except ValueError:
            print("incorrect values")
            continue

    
    cv2.imshow("annotations",img)
    cv2.waitKey(0)

img = cv2.imread(r"D:\Pavan\DataSets\Face_Detection_Dataset\images\train\0a0d7a87378422e3.jpg")
folder_path = r"D:\Pavan\DataSets\Face_Detection_Dataset\images\train"

for f in os.listdir(folder_path):
    if f.lower().endswith(('.jpg','.jpeg','.png')):
        image_path = os.path.join(folder_path,f)
        base_name = os.path.splitext(f)[0]
        text_path = os.path.join(folder_path,base_name+".txt")

        if os.path.exists(image_path) and os.path.exists(text_path):
            check_annotations(image_path,text_path)


gnd_truth = [110,44,600,677] #tlx,tly,brx,bry

#Step 1: Extract Region proposals
proposals = get_region_proposals(img)

#Step 2: Create Positive and Negative samples
positive_set,negative_set,training_labels = get_best_BB_list(proposals,gnd_truth)

outputimg = img.copy()
for box in positive_set:
    cv2.rectangle(outputimg,(box[0],box[1]),(box[2],box[3]),(0,255,0),2)

cv2.imwrite("final_boxes.jpg",outputimg)
print("Done!!..")

# # plt.imshow(img_rz)
# # plt.show()

# #img_lbl,regions = ss.selective_search(img_rz,scale=500,sigma=0.9,min_size=10)

# #candidates = set()
# #https://www.geeksforgeeks.org/computer-vision/opencv-selective-search-for-object-detection/#selective-search


