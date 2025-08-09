# COCO Segmentation JSON --> YOLOv8 Segmentation TXT Format
# Save this as coco_to_yolov8seg.py

import json
import os
from tqdm import tqdm
import cv2

# ---- CONFIG ----
# Change these paths depending on which split you're processing
coco_json_path = r"C:\Users\grace\Dataset\images\test\annotations.json"
images_dir = r"C:\Users\grace\Dataset\images\test"
labels_dir = r"C:\Users\grace\Dataset\labels\test"



# ---- Ensure labels directory exists ----
os.makedirs(labels_dir, exist_ok=True)

# ---- Load COCO JSON ----
with open(coco_json_path, 'r') as f:
    coco = json.load(f)

# ---- Build Lookup Tables ----
image_id_to_filename = {img['id']: img['file_name'] for img in coco['images']}
category_id_to_class_id = {cat['id']: idx for idx, cat in enumerate(coco['categories'])}

# ---- Process Annotations ----
labels = {img['file_name']: [] for img in coco['images']}

for ann in tqdm(coco['annotations']):
    if ann['iscrowd'] == 1:
        continue  # Skip crowd annotations

    image_filename = image_id_to_filename[ann['image_id']]
    segmentation = ann['segmentation'][0]  # Assuming polygon
    category_id = ann['category_id']
    class_id = category_id_to_class_id[category_id]

    # Get bbox for normalization
    bbox = ann['bbox']  # [x, y, width, height]
    x_center = bbox[0] + bbox[2]/2
    y_center = bbox[1] + bbox[3]/2

    # Get image size (we assume filename exists in images_dir)
    img_path = os.path.join(images_dir, image_filename)
    if not os.path.exists(img_path):
        print(f"Warning: Image {img_path} not found, skipping.")
        continue

    img = cv2.imread(img_path)
    height, width = img.shape[:2]

    # Normalize bbox
    x_center /= width
    y_center /= height
    bbox_width = bbox[2] / width
    bbox_height = bbox[3] / height

    # Normalize segmentation points
    norm_seg = []
    for i in range(0, len(segmentation), 2):
        x = segmentation[i] / width
        y = segmentation[i+1] / height
        norm_seg.extend([x, y])

    # YOLOv8-seg label format
    label_line = [str(class_id), f"{x_center:.6f}", f"{y_center:.6f}", f"{bbox_width:.6f}", f"{bbox_height:.6f}"]
    label_line += [f"{p:.6f}" for p in norm_seg]

    labels[image_filename].append(' '.join(label_line))

# ---- Write Labels ----
for img_filename, label_lines in labels.items():
    label_filename = os.path.splitext(img_filename)[0] + ".txt"
    with open(os.path.join(labels_dir, label_filename), 'w') as f:
        f.write('\n'.join(label_lines))

print("Conversion complete! Labels saved in:", labels_dir)

# To process 'valid' split, change:
# coco_json_path = "./valid/annotations.json"
# images_dir = "./valid/images"
# labels_dir = "./labels/valid"

# Similarly for 'test' split.