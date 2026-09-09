"""
Ground-truth target generation + loss functions for CenterNet.

The core trick of CenterNet: instead of matching boxes to anchors,
we render each ground-truth box as a 2D Gaussian "blob" centered on
the box's center point, on a heatmap of size (H/4, W/4). The network
is trained to reproduce that heatmap, plus regress size/offset only
at the exact center pixel.
"""

import math
import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------
# Gaussian heatmap rendering
# --------------------------------------------------------------------

def gaussian_radius(box_size, min_overlap=0.7):
    """
    Given a box's (height, width), compute the radius of a Gaussian such
    that any point within that radius, if used as a "predicted center",
    would still produce a box with at least `min_overlap` IoU with the
    ground truth. This is the formula from the original CenterNet/
    CornerNet papers (solves 3 quadratic equations for 3 corner cases).
    """
    height, width = box_size

    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = math.sqrt(max(b1 ** 2 - 4 * a1 * c1, 0))
    r1 = (b1 - sq1) / (2 * a1)

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = math.sqrt(max(b2 ** 2 - 4 * a2 * c2, 0))
    r2 = (b2 - sq2) / (2 * a2)

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = math.sqrt(max(b3 ** 2 - 4 * a3 * c3, 0))
    r3 = (b3 + sq3) / (2 * a3)

    return max(min(r1, r2, r3), 0)


def draw_gaussian(heatmap, center, radius, k=1.0):
    """Paints a 2D Gaussian bump into `heatmap` at `center`, in-place.
    If two objects' Gaussians overlap, keeps the element-wise max
    (so we don't dilute a strong peak with a weaker nearby one)."""
    diameter = 2 * radius + 1
    sigma = diameter / 6.0
    x = np.arange(-radius, radius + 1)
    y = x[:, None]
    gaussian = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    gaussian[gaussian < np.finfo(gaussian.dtype).eps * gaussian.max()] = 0

    x_c, y_c = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]
    radius = int(radius)

    left, right = min(x_c, radius), min(width - x_c, radius + 1)
    top, bottom = min(y_c, radius), min(height - y_c, radius + 1)

    masked_heatmap = heatmap[y_c - top:y_c + bottom, x_c - left:x_c + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def build_targets(boxes, labels, num_classes, output_size, down_ratio=4):
    """
    Convert one image's ground-truth boxes into CenterNet training targets.

    Args:
        boxes: (N, 4) array of [x1, y1, x2, y2] in ORIGINAL image pixel scale.
        labels: (N,) array of class indices.
        num_classes: total number of classes.
        output_size: (out_h, out_w) — the H/4, W/4 feature map size.
        down_ratio: stride between input image and output feature map.

    Returns dict of numpy arrays:
        heatmap: (num_classes, out_h, out_w)
        offset:  (2, out_h, out_w)      -- only valid at object centers
        size:    (2, out_h, out_w)      -- only valid at object centers
        mask:    (out_h, out_w)         -- 1 at object centers, else 0
        indices: kept for reference (not required by the loss below)
    """
    out_h, out_w = output_size
    heatmap = np.zeros((num_classes, out_h, out_w), dtype=np.float32)
    offset = np.zeros((2, out_h, out_w), dtype=np.float32)
    size = np.zeros((2, out_h, out_w), dtype=np.float32)
    mask = np.zeros((out_h, out_w), dtype=np.float32)

    for box, cls in zip(boxes, labels):
        x1, y1, x2, y2 = box
        w, h = (x2 - x1) / down_ratio, (y2 - y1) / down_ratio
        if w <= 0 or h <= 0:
            continue

        cx, cy = (x1 + x2) / 2 / down_ratio, (y1 + y2) / 2 / down_ratio
        cx_int, cy_int = int(cx), int(cy)
        if not (0 <= cx_int < out_w and 0 <= cy_int < out_h):
            continue

        radius = max(0, int(gaussian_radius((h, w))))
        draw_gaussian(heatmap[int(cls)], (cx_int, cy_int), radius)

        offset[0, cy_int, cx_int] = cx - cx_int   # sub-pixel dx
        offset[1, cy_int, cx_int] = cy - cy_int   # sub-pixel dy
        size[0, cy_int, cx_int] = w
        size[1, cy_int, cx_int] = h
        mask[cy_int, cx_int] = 1.0

    return {"heatmap": heatmap, "offset": offset, "size": size, "mask": mask}


# --------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------

def focal_loss(pred, target, alpha=2.0, beta=4.0):
    """
    Penalty-reduced pixel-wise focal loss (CornerNet/CenterNet variant).

    Standard focal loss handles fg/bg imbalance for a binary target.
    Here the target isn't binary -- it's a soft Gaussian -- so
    non-center pixels close to the true center are penalized LESS than
    pixels far away (the (1-target)^beta term). This stops the network
    from being punished harshly for being "almost right".

    pred, target: (B, C, H, W), both in [0, 1] (pred post-sigmoid).
    """
    pos_mask = target.eq(1).float()
    neg_mask = target.lt(1).float()

    neg_weights = torch.pow(1 - target, beta)

    pred = torch.clamp(pred, min=1e-4, max=1 - 1e-4)  # avoid log(0)

    pos_loss = torch.log(pred) * torch.pow(1 - pred, alpha) * pos_mask
    neg_loss = torch.log(1 - pred) * torch.pow(pred, alpha) * neg_weights * neg_mask

    num_pos = pos_mask.sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        return -neg_loss
    return -(pos_loss + neg_loss) / num_pos


def reg_l1_loss(pred, target, mask):
    """
    L1 loss for size/offset regression, computed ONLY at object-center
    pixels (mask=1 elsewhere is meaningless -- there's no ground truth
    box regression target at a background pixel).

    pred, target: (B, 2, H, W)
    mask: (B, H, W) — 1 at object centers.
    """
    mask = mask.unsqueeze(1).expand_as(pred).float()
    loss = torch.abs(pred * mask - target * mask).sum()
    num_pos = mask.sum() + 1e-4
    return loss / num_pos


class CenterNetLoss(nn.Module):
    """Combined loss: L = L_heatmap + lambda_size * L_size + lambda_offset * L_offset

    Default weights (0.1 for size, 1.0 for offset) are taken directly
    from the original paper's hyperparameters.
    """

    def __init__(self, lambda_size=0.1, lambda_offset=1.0):
        super().__init__()
        self.lambda_size = lambda_size
        self.lambda_offset = lambda_offset

    def forward(self, preds, targets):
        hm_loss = focal_loss(preds["heatmap"], targets["heatmap"])
        off_loss = reg_l1_loss(preds["offset"], targets["offset"], targets["mask"])
        size_loss = reg_l1_loss(preds["size"], targets["size"], targets["mask"])

        total = hm_loss + self.lambda_size * size_loss + self.lambda_offset * off_loss
        return {
            "total_loss": total,
            "heatmap_loss": hm_loss,
            "size_loss": size_loss,
            "offset_loss": off_loss,
        }
