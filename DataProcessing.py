import os
import pandas as pd
import cv2
import numpy as np


# Load the dataset from Excel file
def load_dataset(file_path):
    df = pd.read_excel(file_path, engine='openpyxl')
    return df


# Preprocess and organize the data for each image
def preprocess_data(df, image_folder):
    images = []
    boxes = []
    classes = []

    image_grouped = df.groupby('filename')

    for image_id, group in image_grouped:
        # Load the image
        image_path = os.path.join(image_folder, image_id)
        image = cv2.imread(image_path)

        # Initialize lists for bounding boxes and class labels
        bboxes = []
        class_labels = []

        for _, row in group.iterrows():
            # Collect bounding box data: xmin, ymin, xmax, ymax
            bbox = [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
            bboxes.append(bbox)

            # Convert class to an integer (e.g., Root=0, Trunk=1, Canopy=2)
            class_label = {'Root': 0, 'Trunk': 1, 'Canopy': 2}[row['class']]
            class_labels.append(class_label)

        # Store the processed image, bounding boxes, and class labels
        images.append(image)
        boxes.append(bboxes)
        classes.append(class_labels)

    return np.array(images), boxes, classes


# Paths to the folders
train_folder = 'C:\\Users\\User\Downloads\Tree-Parts.v6-best-model.tensorflow\\train'
valid_folder = 'C:\\Users\\User\Downloads\Tree-Parts.v6-best-model.tensorflow\\valid'
test_folder = 'C:\\Users\\User\Downloads\Tree-Parts.v6-best-model.tensorflow\\valid'

# Load and preprocess the data for each set
train_df = load_dataset(os.path.join(train_folder, 'train_annotations.xlsx'))
valid_df = load_dataset(os.path.join(valid_folder, 'valid_annotations.xlsx'))
test_df = load_dataset(os.path.join(test_folder, 'test_annotations.xlsx'))

train_images, train_bboxes, train_labels = preprocess_data(train_df, train_folder)
valid_images, valid_bboxes, valid_labels = preprocess_data(valid_df, valid_folder)
test_images, test_bboxes, test_labels = preprocess_data(test_df, test_folder)
# Convert lists of bounding boxes and labels into a DataFrame and save as CSV

def save_to_csv(image_ids, bboxes, labels, file_name):
    # Flatten the bounding boxes and labels into a single row per image and each bounding box
    data = []
    for image_id, bbox_list, label_list in zip(image_ids, bboxes, labels):
        for bbox, label in zip(bbox_list, label_list):
            data.append([image_id, *bbox, label])  # Include image_id, xmin, ymin, xmax, ymax, class_label

    # Create a DataFrame and save as CSV
    df = pd.DataFrame(data, columns=['image_id', 'xmin', 'ymin', 'xmax', 'ymax', 'class_label'])
    df.to_csv(file_name, index=False)

# Save train, validation, and test data to CSV files
save_to_csv(train_images, train_bboxes, train_labels, 'train_data.csv')
save_to_csv(valid_images, valid_bboxes, valid_labels, 'valid_data.csv')
save_to_csv(test_images, test_bboxes, test_labels, 'test_data.csv')

print("Data has been saved to CSV files.")