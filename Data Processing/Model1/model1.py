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
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Decode bounding boxes and labels
    xmin = tf.sparse.to_dense(parsed_features['image/object/bbox/xmin'])
    ymin = tf.sparse.to_dense(parsed_features['image/object/bbox/ymin'])
    xmax = tf.sparse.to_dense(parsed_features['image/object/bbox/xmax'])
    ymax = tf.sparse.to_dense(parsed_features['image/object/bbox/ymax'])
    labels = tf.sparse.to_dense(parsed_features['image/object/class/label'])

    # Combine bounding box coordinates
    bboxes = tf.stack([xmin, ymin, xmax, ymax], axis=-1)

    # Pad to ensure there are exactly 3 bounding boxes (for canopy, trunk, root)
    bboxes = tf.pad(bboxes, [[0, 3 - tf.shape(bboxes)[0]], [0, 0]], constant_values=0.0)
    labels = tf.pad(labels, [[0, 3 - tf.shape(labels)[0]]], constant_values=-1)

    return image,  {'bbox_output_reshape': bboxes, 'class_output_reshape': labels}


def masked_sparse_categorical_crossentropy(y_true, y_pred):
    # Extract class labels (last dimension)
    class_labels = y_true[..., 0]  # Assuming the first element is the class label

    # Create a mask where valid labels are marked as True (labels > 0)
    mask = tf.cast(class_labels > 0, tf.float32)

    # Adjust class labels to ignore the "missing" label (set to 0)
    adjusted_y_true = tf.where(class_labels == 0, -1, class_labels)

    # Compute sparse categorical cross-entropy
    loss = tf.keras.losses.sparse_categorical_crossentropy(adjusted_y_true, y_pred, from_logits=False)

    # Apply the mask to the loss
    loss *= mask

    # Average the loss only over valid labels
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)


def load_tfrecord(tfrecord_file):
    dataset = tf.data.TFRecordDataset(tfrecord_file, buffer_size=8 * 1024 * 1024)  # 8 MB
    dataset = dataset.map(parse_tfrecord_fn)
    return dataset


# Load datasets
train_dataset = load_tfrecord('C:\\Users\\User\PycharmProjects\TreeTailsModel\Data Processing\TFRecords\\train.tfrecord').batch(16).shuffle(1000).prefetch(
    tf.data.experimental.AUTOTUNE)
valid_dataset = load_tfrecord('C:\\Users\\User\PycharmProjects\TreeTailsModel\Data Processing\TFRecords\\valid.tfrecord').batch(16).prefetch(
    tf.data.experimental.AUTOTUNE)


# Train the model
def preprocess_image(image, bbox):
    image = tf.image.resize(image, (224, 224))
    return image, bbox


train_dataset = train_dataset.map(preprocess_image)
valid_dataset = valid_dataset.map(preprocess_image)


# 2. Build Model
def create_model():
    base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False)
    x = layers.GlobalAveragePooling2D()(base_model.output)

    # Predict bounding boxes: Output shape (3 parts × 4 coordinates)
    bbox_output = layers.Dense(12, activation='softmax', name='bounding_boxes')(x)  # 3 parts × [xmin, ymin, xmax, ymax]
    bbox_output = layers.Reshape((3, 4), name='bbox_output_reshape')(bbox_output)  # Reshape to (3, 4)

    # Predict class labels: Output shape (3 parts × num_classes)
    num_classes = 4  # Classes: canopy, trunk, root
    class_output = layers.Dense(3 * num_classes, activation='softmax', name='class_labels')(x)
    class_output = layers.Reshape((3, num_classes), name='class_output_reshape')(class_output)  # Reshape to (3, num_classes)

    model = models.Model(inputs=base_model.input, outputs=[bbox_output, class_output])

    model.compile(
        optimizer='adam',
        loss={
            'bbox_output_reshape': 'mse',
            'class_output_reshape': 'sparse_categorical_crossentropy'
        },
        metrics={
            'bbox_output_reshape': 'mae',
            'class_output_reshape': 'accuracy'
        }
    )

    return model


model = create_model()



model.fit(train_dataset, epochs=10, validation_data=valid_dataset)
model.save('./saved_model.keras')
# 4. Post-Processing and Analysis
# Add rules for analyzing detected parts and mapping to personality traits.
# import tensorflow as tf
# from tensorflow.keras import layers, models
#
# # 1. Parse TFRecords
# def parse_tfrecord_fn(example):
#     feature_map = {
#         'image/encoded': tf.io.FixedLenFeature([], tf.string),
#         'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
#         'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
#         'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
#         'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
#         'image/object/class/label': tf.io.VarLenFeature(tf.int64),
#     }
#     parsed_features = tf.io.parse_single_example(example, feature_map)
#
#     # Decode image
#     image = tf.io.decode_jpeg(parsed_features['image/encoded'])
#     image = tf.image.resize(image, [224, 224])
#     image = tf.image.convert_image_dtype(image, tf.float32)  # Normalize image to [0, 1]
#
#     # Decode bounding box coordinates and labels
#     xmin = tf.sparse.to_dense(parsed_features['image/object/bbox/xmin'])
#     ymin = tf.sparse.to_dense(parsed_features['image/object/bbox/ymin'])
#     xmax = tf.sparse.to_dense(parsed_features['image/object/bbox/xmax'])
#     ymax = tf.sparse.to_dense(parsed_features['image/object/bbox/ymax'])
#     labels = tf.sparse.to_dense(parsed_features['image/object/class/label'])
#
#     # Stack bounding boxes into a single tensor (shape: [num_boxes, 4])
#     bboxes = tf.stack([xmin, ymin, xmax, ymax], axis=-1)
#
#     # Return image, bounding boxes, and labels
#     return image, {'bounding_boxes': bboxes, 'class_labels': labels}
#
#
# # def load_tfrecord(tfrecord_file):
# #     dataset = tf.data.TFRecordDataset(tfrecord_file, buffer_size=8 * 1024 * 1024)  # 8 MB
# #     dataset = dataset.map(parse_tfrecord_fn)
# #     return dataset
# #
# # # Load datasets
# # train_dataset = load_tfrecord('../Data Processing/TFRecords/train.tfrecord').shuffle(1000)
# # valid_dataset = load_tfrecord('../Data Processing/TFRecords/valid.tfrecord')
# #
# # # 2. Preprocess Function
# # # 2. Preprocess Function
# # def preprocess_image(image, targets):
# #     bboxes = targets['bounding_boxes']
# #     labels = targets['class_labels']
# #
# #     # Ensure the image has 3 channels (for RGB)
# #     image = tf.ensure_shape(image, (224, 224, 3))
# #
# #     # Keep bounding boxes and labels as ragged tensors
# #     bboxes = tf.RaggedTensor.from_tensor(bboxes)
# #     labels = tf.RaggedTensor.from_tensor(labels)
# #
# #     return image, {'bounding_boxes': bboxes, 'class_labels': labels}
# #
# #
# # # 3. Pad Targets for Batching
# # def pad_targets(image, targets):
# #     bboxes = targets['bounding_boxes'].to_tensor(default_value=0.0)  # Pad bounding boxes
# #     labels = targets['class_labels'].to_tensor(default_value=-1)  # Pad labels
# #
# #     return image, {'bounding_boxes': bboxes, 'class_labels': labels}
# #
# #
# # # 4. Dataset Preparation with Mapping and Padding
# # train_dataset = (train_dataset
# #                  .map(preprocess_image)
# #                  .map(pad_targets)
# #                  .batch(32, drop_remainder=True)
# #                  .prefetch(tf.data.experimental.AUTOTUNE))
# #
# # valid_dataset = (valid_dataset
# #                  .map(preprocess_image)
# #                  .map(pad_targets)
# #                  .batch(32, drop_remainder=True)
# #                  .prefetch(tf.data.experimental.AUTOTUNE))
# #
# #
# # # 3. Build Model
# # def create_model():
# #     base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False)
# #     x = layers.GlobalAveragePooling2D()(base_model.output)
# #
# #     # Predict bounding boxes (10 boxes × 4 coordinates)
# #     bbox_output = layers.Dense(40, activation='sigmoid', name='bounding_boxes')(x)
# #     bbox_output = layers.Reshape((10, 4), name='bounding_boxes_reshape')(bbox_output)  # Reshape to (10, 4)
# #
# #     # Predict class labels (10 boxes × num_classes)
# #     num_classes = 3
# #     class_output = layers.Dense(10 * num_classes, activation='softmax', name='class_labels')(x)
# #     class_output = layers.Reshape((10, num_classes), name='class_labels_reshape')(class_output)  # Reshape to (10, num_classes)
# #
# #     model = models.Model(inputs=base_model.input, outputs=[bbox_output, class_output])
# #     model.compile(
# #         optimizer='adam',
# #         loss={
# #             'bounding_boxes_reshape': 'mse',
# #             'class_labels_reshape': 'sparse_categorical_crossentropy'
# #         },
# #         metrics={
# #             'bounding_boxes_reshape': 'mae',
# #             'class_labels_reshape': 'accuracy'
# #         }
# #     )
# #     return model
# #
# #
# # model = create_model()
# # model.fit(train_dataset, epochs=10, validation_data=valid_dataset)
# # model.save('./saved_model.keras')
#
#
#
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
#
#
# train_dataset = train_dataset.map(preprocess_image).batch(32, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
# valid_dataset = valid_dataset.map(preprocess_image).batch(32, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
#
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
#
#
#
#


#visual
# for images, targets in train_dataset.take(1):
#     print(targets.keys())
#
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# import numpy as np
#
# def visualize_image_with_bboxes(image_batch, bbox_batch, label_batch):
#     """
#     Visualize a batch of images with their bounding boxes and class labels.
#
#     Parameters:
#     - image_batch: A batch of images (e.g., shape [batch_size, height, width, channels]).
#     - bbox_batch: A batch of bounding boxes (e.g., shape [batch_size, 3, 4]).
#     - label_batch: A batch of labels (e.g., shape [batch_size, 3]).
#     """
#     batch_size = image_batch.shape[0]
#     for i in range(batch_size):
#         image = image_batch[i]
#         bboxes = bbox_batch[i]
#         labels = label_batch[i]
#
#         # Convert image to uint8 for display
#         image = (image * 255).astype(np.uint8)
#
#         # Create a plot
#         fig, ax = plt.subplots(1, figsize=(8, 8))
#         ax.imshow(image)
#
#         for bbox, label in zip(bboxes, labels):
#             if label == -1:  # Skip missing parts
#                 continue
#             xmin, ymin, xmax, ymax = bbox
#             width = xmax - xmin
#             height = ymax - ymin
#
#             # Draw the bounding box
#             rect = patches.Rectangle(
#                 (xmin * image.shape[1], ymin * image.shape[0]),  # Scale bbox to image size
#                 width * image.shape[1],
#                 height * image.shape[0],
#                 linewidth=2,
#                 edgecolor='red',
#                 facecolor='none'
#             )
#             ax.add_patch(rect)
#
#             # Annotate with the label
#             part_name = ['Root', 'Trunk', 'Canopy'][label]
#             ax.text(
#                 xmin * image.shape[1],
#                 ymin * image.shape[0] - 5,
#                 part_name,
#                 color='yellow',
#                 fontsize=12,
#                 bbox=dict(facecolor='blue', alpha=0.5)
#             )
#
#         plt.axis('off')
#         plt.show()
#
# # Example usage
# for images, targets in train_dataset.take(1):  # Take one batch
#     visualize_image_with_bboxes(
#         images.numpy(),
#         targets['bbox_output_reshape'].numpy(),
#         targets['class_output_reshape'].numpy()
#     )

@tf.keras.utils.register_keras_serializable()
def iou_metric(y_true, y_pred):
    # Extract coordinates
    xmin_true, ymin_true, xmax_true, ymax_true = tf.split(y_true, 4, axis=-1)
    xmin_pred, ymin_pred, xmax_pred, ymax_pred = tf.split(y_pred, 4, axis=-1)

    # Calculate the intersection
    intersect_xmin = tf.maximum(xmin_true, xmin_pred)
    intersect_ymin = tf.maximum(ymin_true, ymin_pred)
    intersect_xmax = tf.minimum(xmax_true, xmax_pred)
    intersect_ymax = tf.minimum(ymax_true, ymax_pred)
    intersect_area = tf.maximum(0.0, intersect_xmax - intersect_xmin) * tf.maximum(0.0,
                                                                                   intersect_ymax - intersect_ymin)

    # Calculate the union
    true_area = (xmax_true - xmin_true) * (ymax_true - ymin_true)
    pred_area = (xmax_pred - xmin_pred) * (ymax_pred - ymin_pred)
    union_area = true_area + pred_area - intersect_area

    # Compute IoU
    iou = intersect_area / tf.maximum(union_area, 1e-6)
    return tf.reduce_mean(iou)
# 2. Build Model
def create_model():
    base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False)
    x = layers.GlobalAveragePooling2D()(base_model.output)

    # Predict bounding boxes: Output shape (3 parts × 4 coordinates)
    bbox_output = layers.Dense(12, activation=None, name='bounding_boxes')(x)  # 3 parts × [xmin, ymin, xmax, ymax]
    bbox_output = layers.Reshape((3, 4), name='bbox_output_reshape')(bbox_output)  # Reshape to (3, 4)

    # Predict class labels: Output shape (3 parts × num_classes)
    num_classes = 4  # Classes: canopy, trunk, root
    class_output = layers.Dense(3 * num_classes, activation='softmax', name='class_labels')(x)
    class_output = layers.Reshape((3, num_classes), name='class_output_reshape')(class_output)  # Reshape to (3, num_classes)

    model = models.Model(inputs=base_model.input, outputs=[bbox_output, class_output])
    # model.compile(
    #     optimizer='adam',
    #     loss=masked_sparse_categorical_crossentropy,
    #     metrics=['accuracy']
    # )


    model.compile(
        optimizer='adam',
        loss={
            'bbox_output_reshape': 'mse',
            'class_output_reshape': 'sparse_categorical_crossentropy'
        },
        loss_weights={
            'bbox_output_reshape': 0.8,
            'class_output_reshape': 0.2
        },
        metrics={
            'bbox_output_reshape': ['mae', iou_metric],  # Add IoU for bounding boxes
            'class_output_reshape': 'accuracy'
        }
    )