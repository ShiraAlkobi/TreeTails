from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # Use 'yolov8s.pt' for a slightly larger model

# Train the model
model.train(
    data="C:\\Users\\User\PycharmProjects\TreeTailsModel\YOLO_model\data.yaml",  # Path to your data.yaml file
    epochs=10,                 # Number of epochs to train
    batch=16,                  # Batch size
    imgsz=640,                 # Image size (default: 640)
    project='tree_features',   # Project name for logging
    name='yolov8_training',    # Experiment name
    device='cpu'                  # Set device (0 for GPU, 'cpu' for CPU)
)



