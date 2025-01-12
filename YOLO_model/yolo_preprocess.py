import os
import pandas as pd
import ast
from PIL import Image

# Define paths
csv_path = "C:\\Users\\User\PycharmProjects\TreeTailsModel\YOLO_model\Dataset\\test.csv"  # Directory containing train.csv, valid.csv, test.csv
images_dir = "C:\\Users\\User\Desktop\\tree_images_and_annotations\\valid and test images"  # Directory containing all the images
output_dir = "C:\\Users\\User\Desktop\\tree_images_and_annotations\\test_annotation_yolo" # Directory to save YOLO annotations



# Function to convert bounding box to YOLO format
def convert_to_yolo_format(bbox, img_width, img_height):
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) / 2.0 / img_width
    y_center = (y_min + y_max) / 2.0 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    return x_center, y_center, width, height

# Process each CSV file
# for split in ['train', 'valid', 'test']:
#     csv_path = os.path.join(csv_dir, f'{split}.csv')
data = pd.read_csv(csv_path)

for _, row in data.iterrows():
    image_id = row['image_id']
    bboxes = ast.literal_eval(row['bboxes'])  # Convert string to list
    class_labels = ast.literal_eval(row['class_labels'])  # Convert string to list

    # Load image to get dimensions
    image_path = os.path.join(images_dir, image_id)
    with Image.open(image_path) as img:
        img_width, img_height = img.size

    # Create a YOLO annotation file for the image
    yolo_file_path = os.path.join(output_dir, f"{os.path.splitext(image_id)[0]}.txt")
    with open(yolo_file_path, 'w') as f:
        for bbox, label in zip(bboxes, class_labels):
            x_center, y_center, width, height = convert_to_yolo_format(bbox, img_width, img_height)
            f.write(f"{label} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

print("Preprocessing complete! YOLO annotations saved in:", output_dir)
