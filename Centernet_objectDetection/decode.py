"""
Decode CenterNet's raw (heatmap, offset, size) predictions into final
bounding boxes.

CenterNet's headline trick: it does NOT need traditional NMS. Instead
of suppressing overlapping boxes after the fact, it suppresses
overlapping HEATMAP PEAKS using a 3x3 max-pool ("NMS via max-pooling").
If a pixel is not the local maximum in its own 3x3 neighborhood, it's
zeroed out before top-K selection.
"""

import torch
import torch.nn.functional as F


def _nms_max_pool(heatmap, kernel=3):
    """Keep a pixel only if it equals its own 3x3 max-pooled value
    (i.e., it IS the local maximum). Cheap substitute for NMS."""
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heatmap, kernel, stride=1, padding=pad)
    keep = (hmax == heatmap).float()
    return heatmap * keep


def _topk(heatmap, k=100):
    """Get top-k scoring points across the whole (C, H, W) heatmap."""
    batch, cat, height, width = heatmap.size()

    topk_scores, topk_inds = torch.topk(heatmap.view(batch, cat, -1), k)
    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds // width).float()
    topk_xs = (topk_inds % width).float()

    # flatten across classes and take global top-k
    topk_score_flat, topk_ind_flat = torch.topk(topk_scores.view(batch, -1), k)
    topk_classes = (topk_ind_flat // k).float()

    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind_flat).view(batch, k)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind_flat).view(batch, k)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind_flat).view(batch, k)

    return topk_score_flat, topk_inds, topk_classes, topk_ys, topk_xs


def _gather_feat(feat, ind):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    return feat.gather(1, ind)


def _transpose_and_gather_feat(feat, ind):
    # feat: (B, C, H, W) -> (B, H*W, C), then gather at `ind`
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    return _gather_feat(feat, ind)


@torch.no_grad()
def decode_predictions(preds, k=100, score_thresh=0.3, down_ratio=4):
    """
    Args:
        preds: dict with "heatmap" (B,C,H,W), "offset" (B,2,H,W), "size" (B,2,H,W)
        k: max number of detections to consider per image before thresholding.
        score_thresh: final confidence cutoff.
        down_ratio: stride between heatmap and input image (to rescale boxes back).

    Returns: list (length B) of dicts with "boxes" (N,4) [x1,y1,x2,y2] in
             input-image pixel coordinates, "scores" (N,), "labels" (N,).
    """
    heatmap, offset, size = preds["heatmap"], preds["offset"], preds["size"]
    batch = heatmap.size(0)

    heatmap = _nms_max_pool(heatmap)
    scores, inds, classes, ys, xs = _topk(heatmap, k=k)

    offset = _transpose_and_gather_feat(offset, inds)  # (B, k, 2)
    xs = xs.view(batch, k, 1) + offset[:, :, 0:1]
    ys = ys.view(batch, k, 1) + offset[:, :, 1:2]

    size = _transpose_and_gather_feat(size, inds)  # (B, k, 2) -> w, h
    w = size[:, :, 0:1]
    h = size[:, :, 1:2]

    x1 = (xs - w / 2) * down_ratio
    y1 = (ys - h / 2) * down_ratio
    x2 = (xs + w / 2) * down_ratio
    y2 = (ys + h / 2) * down_ratio
    boxes = torch.cat([x1, y1, x2, y2], dim=2)  # (B, k, 4)

    results = []
    for b in range(batch):
        keep = scores[b] > score_thresh
        results.append({
            "boxes": boxes[b][keep].cpu(),
            "scores": scores[b][keep].cpu(),
            "labels": classes[b][keep].long().cpu(),
        })
    return results
