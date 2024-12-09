import tensorflow as tf

def parse_tfrecord(example_proto):
    feature_description = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    }
    return tf.io.parse_single_example(example_proto, feature_description)

raw_dataset = tf.data.TFRecordDataset('valid.tfrecord')


import matplotlib.pyplot as plt
import tensorflow as tf

# Function to decode image
def decode_image(encoded_image):
    return tf.io.decode_jpeg(encoded_image)

# Visualize a parsed record
def visualize_record(parsed_record):
    # Decode the image
    image = decode_image(parsed_record['image/encoded'])
    image = tf.cast(image, tf.uint8).numpy()  # Convert to NumPy array for visualization

    # Extract bounding box data
    xmin = tf.sparse.to_dense(parsed_record['image/object/bbox/xmin']).numpy()
    ymin = tf.sparse.to_dense(parsed_record['image/object/bbox/ymin']).numpy()
    xmax = tf.sparse.to_dense(parsed_record['image/object/bbox/xmax']).numpy()
    ymax = tf.sparse.to_dense(parsed_record['image/object/bbox/ymax']).numpy()
    labels = tf.sparse.to_dense(parsed_record['image/object/class/label']).numpy()

    # Plot the image
    plt.imshow(image)
    ax = plt.gca()
    for i in range(len(xmin)):
        rect = plt.Rectangle(
            (xmin[i] * image.shape[1], ymin[i] * image.shape[0]),
            (xmax[i] - xmin[i]) * image.shape[1],
            (ymax[i] - ymin[i]) * image.shape[0],
            edgecolor='red',
            facecolor='none',
            linewidth=2
        )
        ax.add_patch(rect)
        ax.text(
            xmin[i] * image.shape[1],
            ymin[i] * image.shape[0] - 10,
            f"Label: {labels[i]}",
            color='blue',
            fontsize=12
        )
    plt.show()

# Parse and visualize the first record
for raw_record in raw_dataset.take(3):
    parsed_record = parse_tfrecord(raw_record)
    visualize_record(parsed_record)

