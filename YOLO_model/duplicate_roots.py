import os
import shutil

# Directories
train_images_dir = "C:/Users/User/Desktop/tree_images_and_annotations/train/images"  # Path to training images
train_annotations_dir = "C:/Users/User/Desktop/tree_images_and_annotations/train/labels"  # Path to training annotations (YOLO format)
output_images_dir = "C:/Users/User/Desktop/tree_images_and_annotations/train/images"  # Directory for duplicated images
output_annotations_dir = "C:/Users/User/Desktop/tree_images_and_annotations/train/labels"  # Directory for duplicated annotations

# Ensure output directories exist
os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_annotations_dir, exist_ok=True)

# Class ID for "root"
ROOT_CLASS_ID = 0

# Helper function to check if an annotation file contains roots
def contains_root(annotation_path):
    with open(annotation_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        cls, *_ = map(float, line.strip().split())
        if int(cls) == ROOT_CLASS_ID:
            return True
    return False

# Duplicate images with roots
def duplicate_images_with_roots():
    for img_name in os.listdir(train_images_dir):
        if not img_name.endswith((".jpg", ".png", ".jpeg")):
            continue
        # Paths for the image and annotation
        img_path = os.path.join(train_images_dir, img_name)
        annotation_path = os.path.join(train_annotations_dir, img_name.replace(".jpg", ".txt"))

        # Check if annotation contains roots
        if not os.path.exists(annotation_path):
            #print(f"Skipping {img_name}: Annotation file not found.")
            continue

        if contains_root(annotation_path):
            # Generate a unique name for the duplicate
            duplicate_img_name = f"dup_{img_name}"
            duplicate_annotation_name = duplicate_img_name.replace(".jpg", ".txt")

            # Paths for the duplicated files
            duplicate_img_path = os.path.join(output_images_dir, duplicate_img_name)
            duplicate_annotation_path = os.path.join(output_annotations_dir, duplicate_annotation_name)

            # Copy image and annotation
            shutil.copy(img_path, duplicate_img_path)
            shutil.copy(annotation_path, duplicate_annotation_path)
            #print(f"Duplicated: {img_name}")

# Run the duplication script
duplicate_images_with_roots()
