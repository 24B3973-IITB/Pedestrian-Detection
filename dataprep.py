import os
import re
import xml.etree.ElementTree as ET
from PIL import Image
import random

# =========================
#  Folder Setup
# =========================
base_dir = r"C:\Users\TEJA\Documents\sedrica"
root_dir = os.path.join(base_dir, "dataset")   # original dataset

# VOC Pascal format root OUTSIDE dataset
voc_root = os.path.join(base_dir, "VOCdevkit", "VOC2012")

# Required Pascal VOC folders
out_ann_dir = os.path.join(voc_root, "Annotations")
out_img_dir = os.path.join(voc_root, "JPEGImages")
split_dir   = os.path.join(voc_root, "ImageSets", "Main")

for folder in [out_ann_dir, out_img_dir, split_dir]:
    os.makedirs(folder, exist_ok=True)

print(" Pascal VOC folder structure created under:", voc_root)

# =========================
#  Input Paths
# =========================
img_dir = os.path.join(root_dir, "PNGImages")      # source images (PNG)
ann_dir = os.path.join(root_dir, "Annotation")     # txt annotations

# =========================
#  PNG → JPG Conversion
# =========================
print("Converting PNG → JPG...")
for fname in os.listdir(img_dir):
    if fname.lower().endswith(".png"):
        img_path = os.path.join(img_dir, fname)
        out_path = os.path.join(out_img_dir, fname.replace(".png", ".jpg"))
        if not os.path.exists(out_path):
            img = Image.open(img_path).convert("RGB")
            img.save(out_path, "JPEG")
print(" PNG → JPG done.")

# =========================
#  Function: VOC XML Writer
# =========================
def create_voc_xml(img_filename, img_size, boxes, labels, out_file):
    annotation = ET.Element("annotation")

    ET.SubElement(annotation, "folder").text = "VOC2012"
    ET.SubElement(annotation, "filename").text = img_filename
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(img_size[0])
    ET.SubElement(size, "height").text = str(img_size[1])
    ET.SubElement(size, "depth").text = str(img_size[2])

    for (xmin, ymin, xmax, ymax), label in zip(boxes, labels):
        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = label
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        bbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bbox, "xmin").text = str(xmin)
        ET.SubElement(bbox, "ymin").text = str(ymin)
        ET.SubElement(bbox, "xmax").text = str(xmax)
        ET.SubElement(bbox, "ymax").text = str(ymax)

    tree = ET.ElementTree(annotation)
    tree.write(out_file, encoding="utf-8", xml_declaration=True)

# =========================
#  TXT → VOC XML Conversion
# =========================
bbox_pattern = re.compile(r"Bounding box .*: \((\d+), (\d+)\) - \((\d+), (\d+)\)")

print("Converting TXT → VOC XML...")
kept = 0
for txt_file in os.listdir(ann_dir):
    if not txt_file.endswith(".txt"):
        continue

    txt_path = os.path.join(ann_dir, txt_file)
    with open(txt_path, "r") as f:
        content = f.read()

    boxes, labels = [], []
    for match in bbox_pattern.finditer(content):
        xmin, ymin, xmax, ymax = map(int, match.groups())
        boxes.append((xmin, ymin, xmax, ymax))
        labels.append("person")  # Example: map PASpersonWalking → person

    if not boxes:
        print(f" Skipped {txt_file} (no valid boxes)")
        continue

    img_name = txt_file.replace(".txt", ".jpg")
    img_path = os.path.join(out_img_dir, img_name)
    if not os.path.exists(img_path):
        print(f" Skipped {txt_file} (no matching image)")
        continue

    img = Image.open(img_path)
    w, h = img.size
    c = len(img.getbands())  # RGB or grayscale

    out_file = os.path.join(out_ann_dir, txt_file.replace(".txt", ".xml"))
    create_voc_xml(img_name, (w, h, c), boxes, labels, out_file)
    kept += 1

print(f" Annotation conversion complete. Kept {kept} valid files.")

# =========================
#  Train/Val Split
# =========================
all_ids = [f.replace(".xml", "") for f in os.listdir(out_ann_dir) if f.endswith(".xml")]
random.shuffle(all_ids)
split_idx = int(0.8 * len(all_ids))

train_ids = all_ids[:split_idx]
val_ids = all_ids[split_idx:]

with open(os.path.join(split_dir, "train.txt"), "w") as f:
    f.write("\n".join(train_ids))
with open(os.path.join(split_dir, "val.txt"), "w") as f:
    f.write("\n".join(val_ids))
with open(os.path.join(split_dir, "trainval.txt"), "w") as f:
    f.write("\n".join(all_ids))

print(f" Train/Val split done. Total: {len(all_ids)}, Train: {len(train_ids)}, Val: {len(val_ids)}")

# =========================
#  Quick Check
# =========================
print("\n Quick Check:")
print("JPEGImages count:", len(os.listdir(out_img_dir)))
print("Annotations count:", len(os.listdir(out_ann_dir)))
print("Train.txt entries:", len(train_ids))
print("Val.txt entries:", len(val_ids))
