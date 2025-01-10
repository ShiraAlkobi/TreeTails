import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from model1 import iou_metric
# Load your trained model
model = tf.keras.models.load_model('saved_model.keras', compile=False)
model.trainable = False
# Define function to preprocess the image
def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image)
    input_image = tf.image.resize(image, (224, 224)) / 255.0  # Resize and normalize
    return image, tf.expand_dims(input_image, axis=0)  # Original and batched image

# Visualization function
def visualize_predictions(image_path, predictions):
    original_image, input_image = preprocess_image(image_path)
    height, width = original_image.shape[0], original_image.shape[1]

    predicted_bboxes, predicted_classes = predictions
    predicted_classes = tf.argmax(predicted_classes[0], axis=-1).numpy()

    class_names = ["background", "canopy", "trunk", "root"]  # Define your class labels

    plt.figure(figsize=(10, 10))
    plt.imshow(original_image.numpy().astype(np.uint8))
    ax = plt.gca()

    # Draw predicted bounding boxes
    for i, bbox in enumerate(predicted_bboxes[0]):  # Remove batch dimension
        xmin, ymin, xmax, ymax = bbox
        xmin *= width
        ymin *= height
        xmax *= width
        ymax *= height

        # Draw predicted box
        rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        # Annotate with predicted label
        label = class_names[predicted_classes[i]]
        ax.text(xmin, ymin - 10, label, color='red', fontsize=12, bbox=dict(facecolor='yellow', alpha=0.5, edgecolor='none'))

    plt.axis('off')
    plt.show()

# Path to your test image
test_image_path = "C:\\Users\\User\Desktop\\tree.jpg"  # Update with the correct image path

# Make predictions on the image
_, input_image = preprocess_image(test_image_path)
predictions = model.predict(input_image)

# Visualize the results
visualize_predictions(test_image_path, predictions)
