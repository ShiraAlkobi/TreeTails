from ultralytics import YOLO

# Load the trained YOLOv8 model
model = YOLO("C:\\Users\\User\PycharmProjects\TreeTailsModel\YOLO_model\\tree_features\yolov8_training15\weights\\best.pt")  # Replace with the path to your best model

# # Test on the test dataset
# results = model.predict(source="C:\\Users\\User\Desktop\\tree_images_and_annotations\\test\images", save=True, save_txt=True, conf=0.25)
metrics = model.val(data="C:\\Users\\User\PycharmProjects\TreeTailsModel\YOLO_model\data.yaml")
print(metrics)
