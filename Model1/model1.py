import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Parse TFRecords
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
    image = tf.image.resize(image, [224, 224])
    image = tf.image.convert_image_dtype(image, tf.float32)  # Normalize image to [0, 1]

    # Decode bounding box coordinates and labels
    xmin = tf.sparse.to_dense(parsed_features['image/object/bbox/xmin'])
    ymin = tf.sparse.to_dense(parsed_features['image/object/bbox/ymin'])
    xmax = tf.sparse.to_dense(parsed_features['image/object/bbox/xmax'])
    ymax = tf.sparse.to_dense(parsed_features['image/object/bbox/ymax'])
    labels = tf.sparse.to_dense(parsed_features['image/object/class/label'])

    # Stack bounding boxes into a single tensor (shape: [num_boxes, 4])
    bboxes = tf.stack([xmin, ymin, xmax, ymax], axis=-1)

    # Return image, bounding boxes, and labels
    return image, {'bounding_boxes': bboxes, 'class_labels': labels}


def load_tfrecord(tfrecord_file):
    dataset = tf.data.TFRecordDataset(tfrecord_file, buffer_size=8 * 1024 * 1024)  # 8 MB
    dataset = dataset.map(parse_tfrecord_fn)
    return dataset

# Load datasets
train_dataset = load_tfrecord('../Data Processing/TFRecords/train.tfrecord').shuffle(1000)
valid_dataset = load_tfrecord('../Data Processing/TFRecords/valid.tfrecord')

# 2. Preprocess Function
# 2. Preprocess Function
def preprocess_image(image, targets):
    bboxes = targets['bounding_boxes']
    labels = targets['class_labels']

    # Ensure the image has 3 channels (for RGB)
    image = tf.ensure_shape(image, (224, 224, 3))

    # Keep bounding boxes and labels as ragged tensors
    bboxes = tf.RaggedTensor.from_tensor(bboxes)
    labels = tf.RaggedTensor.from_tensor(labels)

    return image, {'bounding_boxes': bboxes, 'class_labels': labels}


# 3. Pad Targets for Batching
def pad_targets(image, targets):
    bboxes = targets['bounding_boxes'].to_tensor(default_value=0.0)  # Pad bounding boxes
    labels = targets['class_labels'].to_tensor(default_value=-1)  # Pad labels

    return image, {'bounding_boxes': bboxes, 'class_labels': labels}


# 4. Dataset Preparation with Mapping and Padding
train_dataset = (train_dataset
                 .map(preprocess_image)
                 .map(pad_targets)
                 .batch(32, drop_remainder=True)
                 .prefetch(tf.data.experimental.AUTOTUNE))

valid_dataset = (valid_dataset
                 .map(preprocess_image)
                 .map(pad_targets)
                 .batch(32, drop_remainder=True)
                 .prefetch(tf.data.experimental.AUTOTUNE))


# 3. Build Model
def create_model():
    base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False)
    x = layers.GlobalAveragePooling2D()(base_model.output)

    # Predict bounding boxes (10 boxes × 4 coordinates)
    bbox_output = layers.Dense(40, activation='sigmoid', name='bounding_boxes')(x)
    bbox_output = layers.Reshape((10, 4), name='bounding_boxes_reshape')(bbox_output)  # Reshape to (10, 4)

    # Predict class labels (10 boxes × num_classes)
    num_classes = 3
    class_output = layers.Dense(10 * num_classes, activation='softmax', name='class_labels')(x)
    class_output = layers.Reshape((10, num_classes), name='class_labels_reshape')(class_output)  # Reshape to (10, num_classes)

    model = models.Model(inputs=base_model.input, outputs=[bbox_output, class_output])
    model.compile(
        optimizer='adam',
        loss={
            'bounding_boxes_reshape': 'mse',
            'class_labels_reshape': 'sparse_categorical_crossentropy'
        },
        metrics={
            'bounding_boxes_reshape': 'mae',
            'class_labels_reshape': 'accuracy'
        }
    )
    return model


model = create_model()
model.fit(train_dataset, epochs=10, validation_data=valid_dataset)
model.save('./saved_model.keras')



# def preprocess_image(image, targets):
#     bboxes = targets['bounding_boxes']
#     labels = targets['class_labels']
#
#     # Ensure the image has 3 channels (for RGB)
#     image = tf.ensure_shape(image, (224, 224, 3))
#
#     # Convert bounding boxes and labels to Ragged Tensors for handling variable-length inputs
#     bboxes = tf.RaggedTensor.from_tensor(bboxes)
#     labels = tf.RaggedTensor.from_tensor(labels)
#     # # Keep bboxes as RaggedTensor, labels as regular tensor
#     # bboxes = tf.RaggedTensor.from_tensor(bboxes)  # Keep bounding boxes as RaggedTensor
#     return image, {'bounding_boxes': bboxes, 'class_labels': labels}


# train_dataset = train_dataset.map(preprocess_image).batch(32, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
# valid_dataset = valid_dataset.map(preprocess_image).batch(32, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)

# 3. Build Model
# def create_model():
#     base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False)
#     x = layers.GlobalAveragePooling2D()(base_model.output)
#
#     # Predict bounding boxes for all objects
#     bbox_output = layers.Dense(4, activation='sigmoid', name='bounding_boxes')(x)  # Predicts [xmin, ymin, xmax, ymax]
#
#     # Predict class labels for all objects
#     num_classes = 3
#     class_output = layers.Dense(num_classes, activation='softmax', name='class_labels')(x)  # Predicts object classes
#
#     model = models.Model(inputs=base_model.input, outputs=[bbox_output, class_output])
#     model.compile(
#         optimizer='adam',
#         loss={
#             'bounding_boxes': 'mse',
#             'class_labels': 'sparse_categorical_crossentropy'
#         },
#         metrics={
#             'bounding_boxes': 'mae',
#             'class_labels': 'accuracy'
#         }
#     )
#     return model
#
# model = create_model()
# model.fit(train_dataset, epochs=10, validation_data=valid_dataset)
# model.save('./saved_model.keras')




