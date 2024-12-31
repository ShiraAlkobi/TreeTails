import tensorflow as tf

def parse_tfrecord_fn(example):
    feature_map = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    }
    parsed_features = tf.io.parse_single_example(example, feature_map)

    # Decode image
    image = tf.io.decode_jpeg(parsed_features['image/encoded'])
    image = tf.image.resize(image, [1024, 1024])  # Resize to fit Faster R-CNN input size
    image = tf.image.convert_image_dtype(image, tf.float32)  # Normalize image to [0, 1]

    # Decode bounding box coordinates and labels
    xmin = tf.sparse.to_dense(parsed_features['image/object/bbox/xmin'])
    ymin = tf.sparse.to_dense(parsed_features['image/object/bbox/ymin'])
    xmax = tf.sparse.to_dense(parsed_features['image/object/bbox/xmax'])
    ymax = tf.sparse.to_dense(parsed_features['image/object/bbox/ymax'])
    labels = tf.sparse.to_dense(parsed_features['image/object/class/label'])

    # Stack bounding boxes into a single tensor (shape: [num_boxes, 4])
    bboxes = tf.stack([xmin, ymin, xmax, ymax], axis=-1)

    return image, {'bounding_boxes': bboxes, 'class_labels': labels}


def load_tfrecord(tfrecord_file):
    dataset = tf.data.TFRecordDataset(tfrecord_file, buffer_size=8 * 1024 * 1024)  # 8 MB
    dataset = dataset.map(parse_tfrecord_fn)
    return dataset


# Load datasets
train_dataset = load_tfrecord('./Data Processing/TFRecords/train.tfrecord').shuffle(1000)
valid_dataset = load_tfrecord('./Data Processing/TFRecords/valid.tfrecord')
