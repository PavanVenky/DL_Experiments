"""
Minimal COCO-style dataset for CenterNet.

Expects a directory of images plus a COCO-format JSON annotation file
(the standard {"images": [...], "annotations": [...], "categories": [...]}
schema used by COCO, LVIS, and most custom-exported datasets from
tools like CVAT/Roboflow/Label Studio).
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO

from utils import build_targets


class CenterNetDataset(Dataset):
    def __init__(self, img_dir, ann_file, input_size=512, down_ratio=4, augment=True):
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.img_ids = sorted(self.coco.getImgIds())
        self.input_size = input_size
        self.down_ratio = down_ratio
        self.output_size = input_size // down_ratio
        self.augment = augment

        cat_ids = sorted(self.coco.getCatIds())
        self.cat_id_to_label = {cid: i for i, cid in enumerate(cat_ids)}
        self.num_classes = len(cat_ids)

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info["file_name"])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]  # COCO format: x, y, width, height
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_id_to_label[ann["category_id"]])
        boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
        labels = np.array(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64)

        # --- resize image + boxes to a fixed square input_size ---
        scale_x = self.input_size / orig_w
        scale_y = self.input_size / orig_h
        image = cv2.resize(image, (self.input_size, self.input_size))
        if len(boxes) > 0:
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y

        # --- simple augmentation: random horizontal flip ---
        if self.augment and np.random.rand() < 0.5:
            image = image[:, ::-1, :].copy()
            if len(boxes) > 0:
                flipped_x1 = self.input_size - boxes[:, 2]
                flipped_x2 = self.input_size - boxes[:, 0]
                boxes[:, 0], boxes[:, 2] = flipped_x1, flipped_x2

        targets = build_targets(
            boxes, labels, self.num_classes,
            output_size=(self.output_size, self.output_size),
            down_ratio=self.down_ratio,
        )

        image = (image.astype(np.float32) / 255.0 - self.mean) / self.std
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()

        sample = {
            "image": image,
            "heatmap": torch.from_numpy(targets["heatmap"]),
            "offset": torch.from_numpy(targets["offset"]),
            "size": torch.from_numpy(targets["size"]),
            "mask": torch.from_numpy(targets["mask"]),
        }
        return sample


def collate_fn(batch):
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "heatmap": torch.stack([b["heatmap"] for b in batch]),
        "offset": torch.stack([b["offset"] for b in batch]),
        "size": torch.stack([b["size"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
    }
