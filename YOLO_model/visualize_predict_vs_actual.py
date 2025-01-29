import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO


# Paths
test_images_dir = "C:\\Users\\User\Desktop\\tree_images_and_annotations\\test\images"    # Update with your test images path
test_annotations_dir = "C:\\Users\\User\Desktop\\tree_images_and_annotations\\test\labels"   # Ground truth annotations directory
model_path = "C:\\Users\\User\PycharmProjects\TreeTailsModel\YOLO_model\\tree_features\yolov8_training15\weights\\best.pt"  # Path to the trained YOLO model

# Class names (update based on your dataset)
class_names = {0: "root", 1: "trunk", 2: "canopy"}

# Load the YOLO model
model = YOLO(model_path)

# Function to draw bounding boxes
def draw_boxes(image, boxes, labels, color, class_names):
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label_text = class_names[label]
        cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

# Parse YOLO annotation files
def parse_annotation_file(file_path, img_width, img_height):
    boxes = []
    labels = []
    with open(file_path, "r") as f:
        for line in f:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            labels.append(int(class_id))
            # Convert YOLO format (normalized) to pixel format
            x1 = (x_center - width / 2) * img_width
            y1 = (y_center - height / 2) * img_height
            x2 = (x_center + width / 2) * img_width
            y2 = (y_center + height / 2) * img_height
            boxes.append([x1, y1, x2, y2])
    return boxes, labels

# Visualize a few images
annotation_files = os.listdir(test_annotations_dir)[:15]  # Use the first 5 annotation files

for annotation_file in annotation_files:
    # Find the corresponding image
    image_name = os.path.splitext(annotation_file)[0] + ".jpg"
    image_path = os.path.join(test_images_dir, image_name)

    if not os.path.exists(image_path):
        print(f"Image {image_name} not found. Skipping...")
        continue

    # Load the image
    image = cv2.imread(image_path)
    img_height, img_width, _ = image.shape

    # Load ground truth annotations
    annotation_path = os.path.join(test_annotations_dir, annotation_file)
    gt_boxes, gt_labels = parse_annotation_file(annotation_path, img_width, img_height)

    # Get predictions from the model
    results = model.predict(source=image_path, save=False, conf=0.25)
    pred_boxes = []
    pred_labels = []
    for result in results[0].boxes:
        x1, y1, x2, y2 = result.xyxy[0].tolist()
        pred_boxes.append([x1, y1, x2, y2])
        pred_labels.append(int(result.cls.item()))

    # Draw ground truth and predicted bounding boxes
    image_with_gt = draw_boxes(image.copy(), gt_boxes, gt_labels, (0, 255, 0), class_names)  # Green for ground truth
    image_with_preds = draw_boxes(image_with_gt, pred_boxes, pred_labels, (0, 0, 255), class_names)  # Red for predictions

    # Show the image
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image_with_preds, cv2.COLOR_BGR2RGB))
    plt.title(f"Image: {image_name}\nGreen: Ground Truth | Red: Predictions")
    plt.axis("off")
    plt.show()
