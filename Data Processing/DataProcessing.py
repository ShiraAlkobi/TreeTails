import os
import pandas as pd
import numpy as np

# Load the dataset from Excel file
def load_dataset(file_path):
    df = pd.read_excel(file_path, engine='openpyxl')
    return df

# Preprocess and organize the data for each image
def preprocess_data(df):
    image_ids = []  # Store image filenames for reference
    boxes = []
    classes = []

    image_grouped = df.groupby('filename')

    for image_id, group in image_grouped:
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

        # Store the filename as the image_id
        image_ids.append(image_id)
        boxes.append(bboxes)
        classes.append(class_labels)

    return image_ids, boxes, classes

# Paths to the folders
train_folder = 'C:\\Users\\User\\Downloads\\Tree-Parts.v6-best-model.tensorflow\\train'
valid_folder = 'C:\\Users\\User\\Downloads\\Tree-Parts.v6-best-model.tensorflow\\valid'
test_folder = 'C:\\Users\\User\\Downloads\\Tree-Parts.v6-best-model.tensorflow\\valid'

# Load and preprocess the data for each set
train_df = load_dataset(os.path.join(train_folder, 'train_annotations.xlsx'))
valid_df = load_dataset(os.path.join(valid_folder, 'valid_annotations.xlsx'))
test_df = load_dataset(os.path.join(test_folder, 'test_annotations.xlsx'))

train_image_ids, train_bboxes, train_labels = preprocess_data(train_df)
valid_image_ids, valid_bboxes, valid_labels = preprocess_data(valid_df)
test_image_ids, test_bboxes, test_labels = preprocess_data(test_df)

# Save processed data to CSV files
import json

def save_to_csv_combined(image_ids, bboxes, labels, file_name):
    # Flatten the bounding boxes and labels into one row per image
    data = []
    for image_id, bbox_list, label_list in zip(image_ids, bboxes, labels):
        # Convert bounding boxes and labels to JSON strings for storage in a single cell
        bbox_str = json.dumps(bbox_list)
        label_str = json.dumps(label_list)
        data.append([image_id, bbox_str, label_str])

    # Create a DataFrame and save as CSV
    df = pd.DataFrame(data, columns=['image_id', 'bboxes', 'class_labels'])
    df.to_csv(file_name, index=False)

# Save train, validation, and test data to CSV files
save_to_csv_combined(train_image_ids, train_bboxes, train_labels, 'CSV files/train_data_combined.csv')
save_to_csv_combined(valid_image_ids, valid_bboxes, valid_labels, 'CSV files/valid_data_combined.csv')
save_to_csv_combined(test_image_ids, test_bboxes, test_labels, 'CSV files/test_data_combined.csv')

print("Data has been saved to combined CSV files.")

