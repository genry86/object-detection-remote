import torch
import math

# -----------------------------
# Encode/Decode Functions
# -----------------------------
def encode_boxes(gt_boxes, anchors):
    # Вход: gt_boxes: Tensor[B, M, 4], anchors: Tensor[B, M, 4]
    # Выход: deltas: Tensor[B, M, 4]
    cx_deltas = (gt_boxes[..., 0] - anchors[..., 0]) / anchors[..., 2]  # (B, M)
    cy_deltas = (gt_boxes[..., 1] - anchors[..., 1]) / anchors[..., 3]  # (B, M)
    w_deltas = torch.log(gt_boxes[..., 2] / anchors[..., 2])           # (B, M)
    h_deltas = torch.log(gt_boxes[..., 3] / anchors[..., 3])           # (B, M)
    deltas = torch.stack([cx_deltas, cy_deltas, w_deltas, h_deltas], dim=-1)  # (B, M, 4)
    return deltas

def decode_boxes(deltas, anchors):
    # Вход: deltas: Tensor[B, M, 4], anchors: Tensor[B, M, 4]
    # Выход: boxes: Tensor[B, M, 4]
    pred_cx = deltas[..., 0] * anchors[..., 2] + anchors[..., 0]  # (B, M)
    pred_cy = deltas[..., 1] * anchors[..., 3] + anchors[..., 1]  # (B, M)
    pred_w = torch.exp(deltas[..., 2]) * anchors[..., 2]          # (B, M)
    pred_h = torch.exp(deltas[..., 3]) * anchors[..., 3]          # (B, M)
    boxes = torch.stack([pred_cx, pred_cy, pred_w, pred_h], dim=-1)  # (B, M, 4)
    return boxes

# -----------------------------
# Anchor Generator
# -----------------------------
def generate_anchors(feature_map_size=(16,12), base_size=40, ratios=[1.0, 0.5, 2.0], stride=16):
    anchors = []
    fm_h, fm_w = feature_map_size
    for i in range(fm_h):
        for j in range(fm_w):
            cx = (j + 0.5) * stride
            cy = (i + 0.5) * stride
            for ratio in ratios:
                w = base_size * math.sqrt(ratio)
                h = base_size * math.sqrt(ratio)
                anchors.append([cx, cy, w, h])
    return torch.tensor(anchors, dtype=torch.float32)

# -----------------------------
# IoU computation
# -----------------------------
def compute_iou(box1, box2):
    inter_x1 = torch.max(box1[0], box2[0])
    inter_y1 = torch.max(box1[1], box2[1])
    inter_x2 = torch.min(box1[2], box2[2])
    inter_y2 = torch.min(box1[3], box2[3])

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = area1 + area2 - inter_area
    return inter_area / union_area

def cXcYwh_to_x1y1x2y2(boxes: torch.Tensor) -> torch.Tensor:
    """
    Преобразует формат аннотаций из [x_center, y_center, width, height]
    в [x_min, y_min, x_max, y_max].

    boxes: Tensor[B, N, 4] — в формате [cx, cy, w, h]
    return: Tensor[B, N, 4] — в формате [x1, y1, x2, y2]
    """
    cx = boxes[..., 0]
    cy = boxes[..., 1]
    w = boxes[..., 2]
    h = boxes[..., 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)