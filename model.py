import os
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
import torchvision.transforms as T
from torchvision.datasets import VOCDetection
import numpy as np
import json
import cv2
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

# -------------------------
# 1. Dataset Preparation
# -------------------------
voc_root = r"C:\Users\TEJA\Documents\sedrica"
year = "2012"   # MUST match "VOC2012" folder

transform = T.Compose([T.ToTensor()])

train_dataset = VOCDetection(root=voc_root, year=year,
                             image_set="train", download=False,
                             transform=transform)
val_dataset = VOCDetection(root=voc_root, year=year,
                           image_set="val", download=False,
                           transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2,
                                           shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=2,
                                         shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# -------------------------
# 2. Parse VOC Annotations
# -------------------------
def parse_voc_target(target_dict):
    boxes, labels = [], []
    objs = target_dict["annotation"]["object"]
    if isinstance(objs, dict):  # single object
        objs = [objs]

    for obj in objs:
        bbox = obj["bndbox"]
        xmin, ymin, xmax, ymax = map(int, (bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]))
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(1)  # pedestrian = class 1

    return {"boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)}

# -------------------------
# 3. Model Setup
# -------------------------
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn(weights=weights)

num_classes = 2  # background + pedestrian
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
    in_features, num_classes
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# -------------------------
# 4. Training Loop
# -------------------------
optimizer = torch.optim.SGD(model.parameters(), lr=0.005,
                            momentum=0.9, weight_decay=0.0005)

num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images = [img.to(device) for img in images]
        targets = [parse_voc_target(t) for t in targets]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()

    print(f"Epoch {epoch+1}, Avg Loss: {running_loss/len(train_loader):.4f}")

# -------------------------
# 5. Evaluation + Visualization
# -------------------------
def compute_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    box1Area = (box1[2]-box1[0]+1) * (box1[3]-box1[1]+1)
    box2Area = (box2[2]-box2[0]+1) * (box2[3]-box2[1]+1)
    return interArea / float(box1Area + box2Area - interArea + 1e-6)

output_dir = r"C:\Users\TEJA\Documents\sedrica\output"
os.makedirs(output_dir, exist_ok=True)

ious, y_true, y_pred = [], [], []
all_labels, all_scores = [], []  # for PR curve

model.eval()
with torch.no_grad():
    for idx, (images, targets) in enumerate(val_loader):
        images = [img.to(device) for img in images]
        outputs = model(images)

        for i, out in enumerate(outputs):
            gt = parse_voc_target(targets[i])
            gt_boxes = gt["boxes"].cpu().numpy()

            pred_boxes = out["boxes"].cpu().numpy()
            pred_scores = out["scores"].cpu().numpy()

            keep = pred_scores > 0.5
            pred_boxes = pred_boxes[keep]
            pred_scores = pred_scores[keep]

            # Metrics
            for g in gt_boxes:
                match_found = False
                for p, s in zip(pred_boxes, pred_scores):
                    iou = compute_iou(g, p)
                    if iou > 0.5:
                        ious.append(iou)
                        y_true.append(1); y_pred.append(1)
                        all_labels.append(1); all_scores.append(s)
                        match_found = True
                        break
                if not match_found:
                    y_true.append(1); y_pred.append(0)
                    all_labels.append(1); all_scores.append(0.0)

            for p, s in zip(pred_boxes, pred_scores):
                if not any(compute_iou(p, g) > 0.5 for g in gt_boxes):
                    y_true.append(0); y_pred.append(1)
                    all_labels.append(0); all_scores.append(s)

            # Save detections (first 5 only)
            if idx < 5:
               img_np = (images[i].cpu().permute(1,2,0).numpy() * 255).astype(np.uint8)
               img_np = np.ascontiguousarray(img_np)   # ensure OpenCV compatibility
               for (xmin, ymin, xmax, ymax) in pred_boxes.astype(int):
                  cv2.rectangle(img_np, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0,255,0), 2)
               out_path = os.path.join(output_dir, f"detection_{idx}_{i}.jpg")
               cv2.imwrite(out_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

# -------------------------
# 6. Precision–Recall Curve
# -------------------------
precision, recall, thresholds = precision_recall_curve(all_labels, all_scores)
ap_score = average_precision_score(all_labels, all_scores)

plt.figure(figsize=(6,6))
plt.plot(recall, precision, marker='.')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"Precision–Recall Curve (AP = {ap_score:.4f})")
plt.grid(True)
plt.savefig(os.path.join(output_dir, "precision_recall_curve.png"))
plt.close()

# -------------------------
# 7. Metrics
# -------------------------
precision_val = precision_score(y_true, y_pred) if y_true else 0
recall_val = recall_score(y_true, y_pred) if y_true else 0
mean_iou = np.mean(ious) if ious else 0.0
mAP = ap_score  # using sklearn AP instead of proxy

metrics = {
    "Precision": float(precision_val),
    "Recall": float(recall_val),
    "Mean_IoU": float(mean_iou),
    "Average Precision": float(ap_score),
    "mAP": float(mAP)
}

print("Evaluation Metrics:", metrics)

# -------------------------
# 8. Save Model + Metrics
# -------------------------
torch.save(model.state_dict(), os.path.join(output_dir, "fasterrcnn_pedestrian.pth"))
with open(os.path.join(output_dir, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

print(f" Training complete. Model, metrics, detections, and PR curve saved in {output_dir}")
