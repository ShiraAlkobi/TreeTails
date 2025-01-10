from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load the trained YOLOv8 model
model = YOLO("C:\\Users\\User\PycharmProjects\TreeTailsModel\YOLO_model\\tree_features\yolov8_training5\weights\\best.pt")  # Update the path if needed

# Path to the image you want to test
image_path = "C:\\Users\\User\Desktop\\2trees.jpg" # Replace with the actual image path

# Perform inference
results = model(image_path)

# Extract predictions
boxes = results[0].boxes.xyxy.numpy()  # Bounding box coordinates [x_min, y_min, x_max, y_max]
scores = results[0].boxes.conf.numpy()  # Confidence scores
labels = results[0].boxes.cls.numpy()  # Class labels (numeric)

# Class names (adjust based on your dataset's `data.yaml`)
class_names = ["root", "trunk", "canopy"]  # Replace with your class labels if different

# Load the image using OpenCV
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Draw bounding boxes on the image
for i, box in enumerate(boxes):
    x_min, y_min, x_max, y_max = box
    label = int(labels[i])
    score = scores[i]

    # Draw the bounding box
    cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)

    # Add label and confidence score
    text = f"{class_names[label]}: {score:.2f}"
    cv2.putText(image, text, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Visualize the image with bounding boxes
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis("off")
plt.title("YOLOv8 Predictions")
plt.show()
