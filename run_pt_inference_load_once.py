# run_pt_inference_load_once.py
# this inference script is written to run on local computer (not Jetson Nano) and output an image with segmentation masks using a Yolov8n-trained image segmentation model in .pt format
# it loads the model once only to decrease inference time, instead of loading the model everytime it needs to run inference on an image
# prints out debug prints for time taken for model load, number of layers, parameters, gradients and FLOPs in YOLOv8n-seg, time taken for inference+post-process

import time
from ultralytics import YOLO
import cv2
import numpy as np

MODEL_PATH = 'C:\\Users\\grace\\runs\\segment\\train2\\weights\\best.pt' # Replace with .pt model path
IMAGE_PATH = 'C:\\Users\\grace\\Dataset\\images\\train\\Screenshot-2025-07-04-190102_png.rf.2a5c6b255b4beb5f218af7b4ccc36f88.jpg' #Replace with input image path
OUTPUT_PATH = 'C:\\Users\\grace\\output_images\\output_image12.jpg' # Replace with output image path

CLASS_COLORS = [
    (0, 255, 255),
    (0, 255, 0),
    (255, 0, 255),
    (255, 255, 0),
    (255, 0, 0),
    (0, 128, 255),
    (128, 0, 255),
    (0, 255, 128),
    (255, 128, 0)
]

CLASS_NAMES = [
    "Carb",
    "Dairy",
    "Dessert",
    "Drink",
    "Fruit",
    "Meat",
    "Pastry",
    "Sauce",
    "Veg"
]

def darken_color(color, factor=0.5):
    return tuple(max(int(c * factor), 0) for c in color)

def masks_to_polygons(masks_data):
    # Convert masks bitmap to polygons using OpenCV contours
    polygons = []
    for mask in masks_data:
        mask_uint8 = (mask * 255).astype('uint8')
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            polygons.append(largest_contour.reshape(-1, 2))
        else:
            polygons.append(np.array([]))  # Empty polygon if no contour found
    return polygons

def run_inference(model, image_path, output_path):
    start_inf = time.time()
    results = model(image_path)
    end_inf = time.time()

    image = cv2.imread(image_path)
    overlay = image.copy()

    # Try this line instead of .data
    masks_data = results[0].masks.masks.cpu().numpy()
    polygons = masks_to_polygons(masks_data)

    classes = results[0].boxes.cls.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()

    for polygon, cls_val, conf_val in zip(polygons, classes, confidences):
        if polygon.size == 0:
            continue  # Skip empty polygons

        raw_class_id = int(cls_val)
        confidence = float(conf_val)
        class_id = raw_class_id - 1  # Adjust for COCO starting at 1

        if 0 <= class_id < len(CLASS_NAMES):
            class_name = CLASS_NAMES[class_id]
            polygon_color = CLASS_COLORS[class_id % len(CLASS_COLORS)]
            rectangle_color = darken_color(polygon_color, factor=0.5)
            text_color = polygon_color
        else:
            class_name = f"Class {raw_class_id}"
            polygon_color = (255, 255, 255)
            rectangle_color = (150, 150, 150)
            text_color = (255, 255, 255)

        pts = np.array(polygon, dtype=np.int32)
        cv2.fillPoly(overlay, [pts], polygon_color)

        M = cv2.moments(pts)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            x, y, w, h = cv2.boundingRect(pts)
            cx, cy = x + 5, y + 15

        label = f"{class_name} {confidence:.2f}"

        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(overlay,
                      (cx, cy - text_height - baseline),
                      (cx + text_width, cy + baseline),
                      rectangle_color,
                      thickness=-1)
        cv2.putText(overlay, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

    alpha = 0.6
    output = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    cv2.imwrite(output_path, output)

    return end_inf - start_inf


def main():
    print("Loading model...")
    start_load = time.time()
    model = YOLO(MODEL_PATH)
    end_load = time.time()
    print(f"Model loaded in {end_load - start_load:.3f} seconds")

    print("Running inference...")
    inf_time = run_inference(model, IMAGE_PATH, OUTPUT_PATH)
    print(f"Inference + postprocess took {inf_time:.3f} seconds")

    print(f"Output saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
