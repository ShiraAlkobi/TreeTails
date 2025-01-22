import os
import cv2
import albumentations as A
import random

# Directories
train_images_dir = "C:/Users/User/Desktop/tree_images_and_annotations/train/images"  # Path to training images
train_annotations_dir = "C:/Users/User/Desktop/tree_images_and_annotations/train/labels"  # Path to training annotations (in YOLO format)
output_images_dir = "C:/Users/User/Desktop/tree_images_and_annotations/train/images"  # Where to save augmented images
output_annotations_dir = "C:/Users/User/Desktop/tree_images_and_annotations/train/labels"  # Where to save augmented annotations

# Ensure output directories exist
os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_annotations_dir, exist_ok=True)

augmentation = A.Compose(
    [
        # Horizontal flips only
        A.HorizontalFlip(p=0.5),

        # Shift and rotate, but limit to maintain the root at the bottom
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.7),

        # Randomly crop the image
        A.RandomCrop(height=512, width=512, p=0.5),

        # Brightness, contrast, and noise
        A.RandomBrightnessContrast(p=0.8),
        A.GaussNoise(p=0.3),
    ],
    bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"])
)


# Helper function to read YOLO annotation files
def read_yolo_annotation(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    annotations = []
    for line in lines:
        cls, x, y, w, h = map(float, line.strip().split())
        annotations.append([x, y, w, h, int(cls)])
    return annotations

# Helper function to save YOLO annotation files
def save_yolo_annotation(file_path, annotations):
    with open(file_path, "w") as f:
        for ann in annotations:
            cls, x, y, w, h = ann
            f.write(f"{cls} {x} {y} {w} {h}\n")

# Normalize bounding boxes for Albumentations
def normalize_bboxes(bboxes, rows, cols):
    normalized = []
    for bbox in bboxes:
        x, y, w, h = bbox
        x_center = x / cols
        y_center = y / rows
        width = w / cols
        height = h / rows
        normalized.append([x_center, y_center, width, height])
    return normalized

# Denormalize bounding boxes for saving
def denormalize_bboxes(bboxes, rows, cols):
    denormalized = []
    for bbox in bboxes:
        x_center, y_center, width, height = bbox
        x = x_center * cols
        y = y_center * rows
        w = width * cols
        h = height * rows
        denormalized.append([x, y, w, h])
    return denormalized

# Clamp bounding box values to [0, 1] range
def clamp_bboxes(bboxes):
    clamped = []
    for bbox in bboxes:
        x, y, w, h = bbox
        x = max(0, min(1, x))
        y = max(0, min(1, y))
        w = max(0, min(1, w))
        h = max(0, min(1, h))
        if w > 0.01 and h > 0.01:  # Filter out very small boxes
            clamped.append([x, y, w, h])
    return clamped

# Apply augmentations and save augmented data
def augment_training_dataset():
    for img_name in os.listdir(train_images_dir):
        if not img_name.endswith((".jpg", ".png", ".jpeg")):
            continue
        # Image and annotation paths
        img_path = os.path.join(train_images_dir, img_name)
        annotation_path = os.path.join(train_annotations_dir, img_name.replace(".jpg", ".txt"))

        # Read image and annotations
        image = cv2.imread(img_path)
        h, w, _ = image.shape
        bboxes = read_yolo_annotation(annotation_path)
        class_labels = [ann[-1] for ann in bboxes]
        bboxes = [ann[:4] for ann in bboxes]

        # Normalize bounding boxes for Albumentations
        bboxes = normalize_bboxes(bboxes, rows=h, cols=w)

        # Apply augmentation
        augmented = augmentation(image=image, bboxes=bboxes, class_labels=class_labels)

        # Get augmented image and bounding boxes
        augmented_image = augmented["image"]
        augmented_bboxes = denormalize_bboxes(augmented["bboxes"], rows=h, cols=w)
        augmented_labels = augmented["class_labels"]

        # Clamp bounding boxes to [0, 1] and filter out invalid ones
        augmented_bboxes = clamp_bboxes(augmented_bboxes)

        if len(augmented_bboxes) == 0:
            print(f"Skipping {img_name}: No valid bounding boxes after augmentation.")
            continue

        # Save augmented image
        aug_img_name = f"aug_{random.randint(1000, 9999)}_{img_name}"
        aug_img_path = os.path.join(output_images_dir, aug_img_name)
        cv2.imwrite(aug_img_path, augmented_image)

        # Save augmented annotations
        aug_annotation_path = os.path.join(output_annotations_dir, aug_img_name.replace(".jpg", ".txt"))
        augmented_annotations = [[label] + bbox for label, bbox in zip(augmented_labels, augmented_bboxes)]
        save_yolo_annotation(aug_annotation_path, augmented_annotations)


# Run augmentation
augment_training_dataset()
