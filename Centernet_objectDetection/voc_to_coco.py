"""
Convert Pascal VOC XML annotations (as used by the Kaggle "Face Mask
Detection" dataset by andrewmvd, and most similar Kaggle detection
datasets) into a single COCO-format JSON file that dataset.py can load
directly via pycocotools.

Expected input layout (this is exactly how the Kaggle dataset unzips):

    face-mask-detection/
        images/
            maksssksksss0.png
            maksssksksss1.png
            ...
        annotations/
            maksssksksss0.xml
            maksssksksss1.xml
            ...

Each XML file follows the standard Pascal VOC schema:

    <annotation>
        <filename>maksssksksss0.png</filename>
        <size><width>512</width><height>366</height><depth>3</depth></size>
        <object>
            <name>with_mask</name>
            <bndbox>
                <xmin>79</xmin><ymin>105</ymin>
                <xmax>109</xmax><ymax>142</ymax>
            </bndbox>
        </object>
        ...
    </annotation>

Usage:
    python voc_to_coco.py \
        --images_dir ./face-mask-detection/images \
        --annotations_dir ./face-mask-detection/annotations \
        --output ./face-mask-detection/annotations_coco.json

    # Optional: also split into train/val JSONs in one go
    python voc_to_coco.py \
        --images_dir ./face-mask-detection/images \
        --annotations_dir ./face-mask-detection/annotations \
        --output ./face-mask-detection/annotations_coco.json \
        --val_split 0.15 --seed 42
"""

import argparse
import json
import os
import random
import xml.etree.ElementTree as ET

from PIL import Image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images_dir", type=str, required=True,
                   help="Folder containing the image files.")
    p.add_argument("--annotations_dir", type=str, required=True,
                   help="Folder containing the matching Pascal VOC .xml files.")
    p.add_argument("--output", type=str, required=True,
                   help="Path to write the COCO JSON file to. "
                        "If --val_split > 0, this is used as the TRAIN output path "
                        "and a sibling '<name>_val.json' is also written.")
    p.add_argument("--val_split", type=float, default=0.0,
                   help="Fraction of images to hold out for validation (e.g. 0.15). "
                        "0 = write a single combined JSON, no split.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def parse_voc_xml(xml_path):
    """Parse one Pascal VOC XML file into (filename, width, height, list of (class_name, x1, y1, x2, y2))."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.find("filename").text

    size_node = root.find("size")
    width = int(size_node.find("width").text)
    height = int(size_node.find("height").text)

    boxes = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        bnd = obj.find("bndbox")
        x1 = float(bnd.find("xmin").text)
        y1 = float(bnd.find("ymin").text)
        x2 = float(bnd.find("xmax").text)
        y2 = float(bnd.find("ymax").text)
        boxes.append((name, x1, y1, x2, y2))

    return filename, width, height, boxes


def build_coco_dict(xml_files, images_dir, categories):
    """categories: dict mapping class_name -> category_id (1-indexed, COCO convention)."""
    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": cid, "name": name} for name, cid in categories.items()
        ],
    }

    ann_id = 1
    for img_id, xml_path in enumerate(xml_files, start=1):
        filename, width, height, boxes = parse_voc_xml(xml_path)

        # Guard against XML/image size mismatches (rare but happens in
        # hand-annotated Kaggle sets) by trusting the actual image file
        # if it's readable, falling back to the XML's stated size.
        img_path = os.path.join(images_dir, filename)
        if os.path.exists(img_path):
            with Image.open(img_path) as im:
                real_w, real_h = im.size
            if (real_w, real_h) != (width, height):
                width, height = real_w, real_h

        coco["images"].append({
            "id": img_id,
            "file_name": filename,
            "width": width,
            "height": height,
        })

        for name, x1, y1, x2, y2 in boxes:
            if name not in categories:
                continue  # skip unexpected/typo'd class names rather than crash
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                continue
            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": categories[name],
                "bbox": [x1, y1, w, h],  # COCO format: [x, y, width, height]
                "area": w * h,
                "iscrowd": 0,
            })
            ann_id += 1

    return coco


def main():
    args = parse_args()

    xml_files = sorted(
        os.path.join(args.annotations_dir, f)
        for f in os.listdir(args.annotations_dir)
        if f.lower().endswith(".xml")
    )
    if not xml_files:
        raise FileNotFoundError(f"No .xml files found in {args.annotations_dir}")
    print(f"Found {len(xml_files)} annotation files.")

    # First pass: discover the full set of class names actually present,
    # so category IDs are assigned consistently regardless of dataset variant.
    class_names = set()
    for xml_path in xml_files:
        _, _, _, boxes = parse_voc_xml(xml_path)
        for name, *_ in boxes:
            class_names.add(name)
    class_names = sorted(class_names)
    categories = {name: i + 1 for i, name in enumerate(class_names)}  # COCO ids start at 1
    print(f"Discovered {len(categories)} classes: {categories}")

    if args.val_split > 0:
        random.seed(args.seed)
        shuffled = xml_files[:]
        random.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * args.val_split))
        val_files = shuffled[:n_val]
        train_files = shuffled[n_val:]

        train_coco = build_coco_dict(train_files, args.images_dir, categories)
        val_coco = build_coco_dict(val_files, args.images_dir, categories)

        with open(args.output, "w") as f:
            json.dump(train_coco, f)
        val_output = args.output.replace(".json", "_val.json")
        with open(val_output, "w") as f:
            json.dump(val_coco, f)

        print(f"Train: {len(train_files)} images, {len(train_coco['annotations'])} boxes -> {args.output}")
        print(f"Val:   {len(val_files)} images, {len(val_coco['annotations'])} boxes -> {val_output}")
    else:
        coco = build_coco_dict(xml_files, args.images_dir, categories)
        with open(args.output, "w") as f:
            json.dump(coco, f)
        print(f"Wrote {len(xml_files)} images, {len(coco['annotations'])} boxes -> {args.output}")


if __name__ == "__main__":
    main()
