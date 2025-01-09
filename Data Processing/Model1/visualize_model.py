import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Load the saved model
model = tf.keras.models.load_model('./saved_model.keras')

# Load and preprocess a test image
def load_and_preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image)
    original_image = tf.image.resize(image, (224, 224)) / 255.0  # Resize and normalize
    input_image = tf.expand_dims(original_image, axis=0)  # Add batch dimension
    return input_image, image

# Visualize predictions
def visualize_predictions(image_path):
    input_image, original_image = load_and_preprocess_image(image_path)

    # Make predictions
    predictions = model.predict(input_image)
    bbox_predictions, class_predictions = predictions

    # Extract bounding boxes and classes
    bbox_predictions = bbox_predictions[0]  # Remove batch dimension
    class_predictions = class_predictions[0]  # Remove batch dimension
    predicted_classes = tf.argmax(class_predictions, axis=-1).numpy()

    # Map class indices to names (e.g., canopy, trunk, root)
    class_names = ["background", "canopy", "trunk", "root"]

    # Plot the image
    plt.figure(figsize=(8, 8))
    plt.imshow(original_image.numpy().astype(np.uint8))
    ax = plt.gca()

    for i, bbox in enumerate(bbox_predictions):
        xmin, ymin, xmax, ymax = bbox

        # Scale coordinates back to the original image size
        width, height = original_image.shape[1], original_image.shape[0]
        xmin *= width
        ymin *= height
        xmax *= width
        ymax *= height

        # Draw bounding box
        rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                         linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        # Annotate with class label
        label = class_names[predicted_classes[i]]
        ax.text(xmin, ymin - 10, label, color='red', fontsize=12,
                bbox=dict(facecolor='yellow', alpha=0.5, edgecolor='none'))

    plt.axis('off')
    plt.show()

# Use the function with your test image
test_image_path = "C:\\Users\\User\Downloads\Tree-Parts.v6-best-model.tensorflow\\valid\\tree_509_png_jpg.rf.a7e44963211cbf7d204f1193546aa8e5.jpg"  # Replace with your test image path
visualize_predictions(test_image_path)
