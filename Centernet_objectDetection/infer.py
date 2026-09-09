"""
Run inference with a trained CenterNet checkpoint on a single image
and save a visualization with drawn boxes.

Usage:
    python infer.py --checkpoint ./checkpoints/centernet_epoch49.pth \
                     --image ./test.jpg --input_size 512 --score_thresh 0.3
"""

import argparse
import cv2
import numpy as np
import torch

from model import CenterNet
from decode import decode_predictions


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--input_size", type=int, default=512)
    p.add_argument("--score_thresh", type=float, default=0.3)
    p.add_argument("--output", type=str, default="output.jpg")
    p.add_argument("--class_names", type=str, default=None,
                   help="Optional comma-separated list of class names, in label-index order.")
    return p.parse_args()


def preprocess(image_path, input_size):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image.shape[:2]

    resized = cv2.resize(image, (input_size, input_size))
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normed = (resized.astype(np.float32) / 255.0 - mean) / std
    tensor = torch.from_numpy(normed.transpose(2, 0, 1)).float().unsqueeze(0)

    scale_x = orig_w / input_size
    scale_y = orig_h / input_size
    return tensor, image, (scale_x, scale_y)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = CenterNet(num_classes=ckpt["num_classes"], backbone_name=ckpt.get("backbone", "resnet18"), pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    tensor, orig_image, (scale_x, scale_y) = preprocess(args.image, args.input_size)
    tensor = tensor.to(device)

    with torch.no_grad():
        preds = model(tensor)
    results = decode_predictions(preds, k=100, score_thresh=args.score_thresh)[0]

    class_names = args.class_names.split(",") if args.class_names else None

    boxes = results["boxes"].numpy()
    scores = results["scores"].numpy()
    labels = results["labels"].numpy()

    vis = cv2.cvtColor(orig_image, cv2.COLOR_RGB2BGR)
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box
        # rescale from model input space back to original image space
        x1, x2 = x1 * scale_x, x2 * scale_x
        y1, y2 = y1 * scale_y, y2 * scale_y
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        label_str = class_names[label] if class_names else str(int(label))
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, f"{label_str} {score:.2f}", (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imwrite(args.output, vis)
    print(f"Detections: {len(boxes)}. Saved visualization to {args.output}")


if __name__ == "__main__":
    main()
