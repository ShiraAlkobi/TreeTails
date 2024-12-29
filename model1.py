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
train_dataset = load_tfrecord('./Data Processing/TFRecords/train.tfrecord').shuffle(1000)
valid_dataset = load_tfrecord('./Data Processing/TFRecords/valid.tfrecord')


# 2. Preprocess Function
def preprocess_image(image, targets):
    bboxes = targets['bounding_boxes']
    labels = targets['class_labels']

    # Ensure the image has 3 channels (for RGB)
    image = tf.ensure_shape(image, (224, 224, 3))

    # Return image, bounding boxes, and labels as ragged tensors
    return image, {
        'bounding_boxes': tf.RaggedTensor.from_tensor(bboxes),
        'class_labels': tf.RaggedTensor.from_tensor(labels)
    }


train_dataset = train_dataset.map(preprocess_image).batch(32, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
valid_dataset = valid_dataset.map(preprocess_image).batch(32, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)


# 3. Build Model
def create_model():
    base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False)
    x = layers.GlobalAveragePooling2D()(base_model.output)

    # Predict bounding boxes for all objects
    bbox_output = layers.Dense(4, activation='sigmoid', name='bounding_boxes')(x)  # Predicts [xmin, ymin, xmax, ymax]

    # Predict class labels for all objects
    num_classes = 3
    class_output = layers.Dense(num_classes, activation='softmax', name='class_labels')(x)  # Predicts object classes

    model = models.Model(inputs=base_model.input, outputs=[bbox_output, class_output])  # Pass the outputs correctly
    model.compile(
        optimizer='adam',
        loss={
            'bounding_boxes': 'mse',
            'class_labels': 'sparse_categorical_crossentropy'
        },
        metrics={
            'bounding_boxes': 'mae',
            'class_labels': 'accuracy'
        }
    )
    return model


model = create_model()
model.fit(train_dataset, epochs=10, validation_data=valid_dataset)
model.save('./saved_model.keras')

# 4. Post-Processing and Analysis
# Add rules for analyzing detected parts and mapping to personality traits.
