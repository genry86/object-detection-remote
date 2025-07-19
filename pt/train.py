import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import os
from torchvision.ops import box_iou

from dataset import RemoteDataset
from model import SSDMobileNet
from utils import encode_boxes, cXcYwh_to_x1y1x2y2

# ------------------------
# Device setup
# ------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ------------------------
# Augmentations
# ------------------------
train_transform = transforms.Compose([
    transforms.Resize((2016, 1512)),  # (height, width)
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

val_transform = transforms.Compose([
    transforms.Resize((2016, 1512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# ------------------------
# Dataset loading
# ------------------------
BASE_DIR = "../"
dataset_path = os.path.join(BASE_DIR, "dataset/train.json")
dataset = RemoteDataset(
    json_path=dataset_path,
    root_dir="..",
    transforms=train_transform
)

STOP_COUNTER_LIMIT = 10
BATCH_SIZE = 4
val_size = int(0.1 * len(dataset))
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# ------------------------
# Model, optimizer, losses
# ------------------------
model = SSDMobileNet(num_classes=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
cls_loss_fn = nn.CrossEntropyLoss()
bbox_loss_fn = nn.SmoothL1Loss()

# ------------------------
# Обучение
# ------------------------
checkpoint_path = os.path.join(BASE_DIR, "training_model", "last.pth")
best_model_path = os.path.join(BASE_DIR, "training_model", "best.pth")
num_epochs = 300
best_val_loss = None
stop_counter = 0
MODEL_PATH_FOLDER = os.path.join(BASE_DIR, "training_model")
os.makedirs(MODEL_PATH_FOLDER, exist_ok=True)
start_epoch = 0

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint.get('best_val_loss', None)
    stop_counter = checkpoint.get('stop_counter', 0)
    print(f"✅ Loaded checkpoint from epoch {start_epoch}, best_val_loss: {best_val_loss:.4f}")

for epoch in range(start_epoch, num_epochs):
    model.train()
    train_loss = 0.0

    def match_targets_with_anchors(anchors, targets):
        gt_labels_list = [t['labels'].to(device) for t in targets]
        gt_boxes_list = [t['boxes'].to(device) // 2 for t in targets]

        B, ANCHORS_NUM, _ = anchors.shape
        matched_labels = torch.zeros((B, ANCHORS_NUM), dtype=torch.long, device=device)  # Tensor[B, ANCHORS_NUM]
        matched_gt_boxes = torch.zeros((B, ANCHORS_NUM, 4), dtype=torch.float32, device=device)  # Tensor[B, ANCHORS_NUM, 4]

        normalized_anchors = cXcYwh_to_x1y1x2y2(anchors)
        for i in range(B):
            normalized_anchor = normalized_anchors[i]

            gt_labels = gt_labels_list[i]
            gt_boxes = gt_boxes_list[i]
            normalized_gt_boxes = cXcYwh_to_x1y1x2y2(gt_boxes)

            ious = box_iou(normalized_anchor, normalized_gt_boxes)  # Tensor[M, N]
            best_ious, best_gt_idx = ious.max(dim=1)                # Tensor[M], Tensor[M]

            positive_mask = best_ious > 0.26
            positive_indices = torch.nonzero(positive_mask).squeeze(1)

            for j in positive_indices:
                anchor_idx = j.item()
                gt_idx = best_gt_idx[anchor_idx].item()  # Always "0" if "single Remote obj"

                matched_labels[i, anchor_idx] = gt_labels[gt_idx]   # Always 1 if "single Remote obj"
                matched_gt_boxes[i, anchor_idx] = gt_boxes[gt_idx]

        return matched_labels, matched_gt_boxes

    loop = tqdm(train_loader, desc=f"", leave=False)
    for batch in loop:
        images, targets = batch                     # Tuple[List[Tensor[3, H, W]], List[Dict{'boxes': FloatTensor[N, 4], 'labels': LongTensor[N]}]]
        images = torch.stack(images, dim=0).to(device)  # Tensor[B, 3, 2016, 1512] -> Tensor[B, 3, 2016, 1512] на устройстве

        cls_logits, boxes_pred, bbox_deltas, anchors = model(images)    # cls_logits: Tensor[B, M, 2], boxes_pred: Tensor[B, M, 4], bbox_deltas: Tensor[B, M, 4], anchors: Tensor[B, M, 4]

        matched_labels, matched_gt_boxes = match_targets_with_anchors(anchors, targets)

        # Переводим в bbox дельты
        target_deltas = encode_boxes(matched_gt_boxes, anchors)  # Tensor[B, ANCHORS_NUM, 4]

        # Вычисление лосса
        preds_classes = cls_logits.view(-1, 2)  # Tensor[4608, 2]
        targets_classes = matched_labels.view(-1)  # Tensor[4608]  [0 or 1] used for index in cls_logits to compare
        cls_loss = cls_loss_fn(preds_classes, targets_classes)  # Scalar

        positive_mask = targets_classes > 0     # to filter boxes loss calculation

        preds_deltas = bbox_deltas.view(-1, 4)              # Tensor[4608, 4]
        targets_deltas = target_deltas.view(-1, 4)          # Tensor[4608, 4]
        box_loss = bbox_loss_fn(preds_deltas[positive_mask], targets_deltas[positive_mask])  # Scalar

        total_loss = cls_loss + box_loss  # Scalar

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        train_loss += total_loss.item()
        loop.set_description(f"Train Epoch [{epoch+1}/{num_epochs}]")
        loop.set_postfix(cls_loss=cls_loss.item(), box_loss=box_loss.item())

    avg_train_loss = train_loss / len(train_loader)

    # Validation (as before)
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        loop = tqdm(val_loader, desc=f"", leave=False)
        for batch in loop:
            images, targets = batch  # Tuple[List[Tensor[3, H, W]], List[Dict{'boxes': Tensor[N, 4], 'labels': Tensor[N]}]]
            images = torch.stack(images, dim=0).to(device)  # Tensor[B, 3, H, W]

            cls_logits, boxes_pred, bbox_deltas, anchors = model(images)  # все: Tensor[B, M, ...]

            matched_labels, matched_gt_boxes = match_targets_with_anchors(anchors, targets)

            target_deltas = encode_boxes(matched_gt_boxes, anchors)  # Tensor[B, M, 4]

            # Class Loss
            preds_classes = cls_logits.view(-1, 2)  # Tensor[B*M, 2]
            targets_classes = matched_labels.view(-1)  # Tensor[B*M]
            cls_loss = cls_loss_fn(preds_classes, targets_classes)

            positive_mask = targets_classes > 0

            # Boxes Loss
            preds_deltas = bbox_deltas.view(-1, 4)  # Tensor[B*M, 4]
            targets_deltas = target_deltas.view(-1, 4)  # Tensor[B*M, 4]
            box_loss = bbox_loss_fn(preds_deltas[positive_mask], targets_deltas[positive_mask])

            total_loss = cls_loss + box_loss
            val_loss += total_loss.item()

            loop.set_description(f"Val Epoch [{epoch + 1}/{num_epochs}]")
            loop.set_postfix(cls_loss=cls_loss.item(), box_loss=box_loss.item())

    avg_val_loss = val_loss / len(val_loader)
    scheduler.step(avg_val_loss)

    if best_val_loss is None or avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        stop_counter = 0
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model saved - epoch - {epoch + 1}, avg_val_loss - {avg_val_loss}")
    else:
        print(f"stop_counter - {stop_counter}, avg_val_loss - {avg_val_loss}")

    # ------------------------
    # Сохранение моделей
    # ------------------------
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
        'stop_counter': stop_counter
    }

    torch.save(state, checkpoint_path)

    # if stop_counter > STOP_COUNTER_LIMIT:
    #     print(f"Training is finished - epoch - {epoch + 1}, with stop counter - {stop_counter}")
    #     break

    stop_counter += 1