import os
import tensorflow as tf
import pandas as pd
import json
from PIL import Image

# Function to create a TFRecord example
def create_tfrecord_example(image_path, bbox_list, label_list):
    # Read and encode the image
    with tf.io.gfile.GFile(image_path, 'rb') as img_file:
        encoded_image = img_file.read()
    image = Image.open(image_path)
    width, height = image.size

    # Serialize bounding boxes and labels (including missing parts)
    xmins = [bbox[0] / width if bbox != [0, 0, 0, 0] else 0.0 for bbox in bbox_list]
    ymins = [bbox[1] / height if bbox != [0, 0, 0, 0] else 0.0 for bbox in bbox_list]
    xmaxs = [bbox[2] / width if bbox != [0, 0, 0, 0] else 0.0 for bbox in bbox_list]
    ymaxs = [bbox[3] / height if bbox != [0, 0, 0, 0] else 0.0 for bbox in bbox_list]
    classes = label_list  # Include all labels, including 0 for missing parts

    # Create a TFRecord Example
    feature = {
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_image])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),
        'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=classes))
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


# Function to write TFRecord file
def write_tfrecord(csv_file, image_folder, output_path):
    df = pd.read_csv(csv_file)
    with tf.io.TFRecordWriter(output_path) as writer:
        for _, row in df.iterrows():
            image_path = os.path.join(image_folder, row['image_id'])
            bbox_list = json.loads(row['bboxes'])
            label_list = json.loads(row['class_labels'])
            tf_example = create_tfrecord_example(image_path, bbox_list, label_list)
            writer.write(tf_example.SerializeToString())

# Paths
train_csv = 'C:\\Users\\User\PycharmProjects\TreeTailsModel\Data Processing\CSV files\\train_data.csv'
valid_csv = 'C:\\Users\\User\PycharmProjects\TreeTailsModel\Data Processing\CSV files\\valid_data.csv'
test_csv = 'C:\\Users\\User\PycharmProjects\TreeTailsModel\Data Processing\CSV files\\test_data.csv'

train_folder = 'C:\\Users\\User\Downloads\Tree-Parts.v6-best-model.tensorflow\\train'
valid_folder = 'C:\\Users\\User\Downloads\Tree-Parts.v6-best-model.tensorflow\\valid'
test_folder = 'C:\\Users\\User\Downloads\Tree-Parts.v6-best-model.tensorflow\\valid'

# Output TFRecord paths
train_tfrecord = 'train.tfrecord'
valid_tfrecord = 'valid.tfrecord'
test_tfrecord = 'test.tfrecord'

# Generate TFRecord files
write_tfrecord(train_csv, train_folder, train_tfrecord)
write_tfrecord(valid_csv, valid_folder, valid_tfrecord)
write_tfrecord(test_csv, test_folder, test_tfrecord)

print("TFRecord files created successfully.")
