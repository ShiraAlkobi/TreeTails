import app
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
from analyze_ouput import parse_yolo_output, analyze_tree_proportions, analyze_feature_proportions, \
    analyze_tree_location, analyze_tree_shapes, generate_personality_output
from flask_cors import CORS
import json


app = Flask(__name__)
CORS(app)

# Define the folder to store uploaded images
app.config['UPLOAD_FOLDER'] = r"C:\Users\User\Desktop"

# Allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Endpoint to upload image
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Process the image and run the model
        # This is where you integrate the YOLO output and your analysis
        image_width, image_height = 800, 600  # Replace with actual image dimensions
        yolo_output = run_yolo_model(file_path)  # Function to run your YOLO model and get output
        features = parse_yolo_output(yolo_output, image_width, image_height)

        # Run analysis
        # indicator = load_indicator()  # Load your indicator data (the JSON file)
        # Open and read the JSON file
        with open('indicator.json', 'r', encoding="utf-8") as file:
            indicator = json.load(file)
        personality_output = generate_personality_output(features, image_width, image_height, indicator)

        return jsonify({'personality_traits': personality_output}), 200
    else:
        return jsonify({'error': 'Invalid file format'}), 400


# Function to load the indicator data (your JSON)
def load_indicator():
    # Return the data from your "indicator" file
    return {
        "tree_proportions": {...},  # Add your actual JSON data here
        "tree_location": {...},
        "canopy": {...},
        "trunk": {...},
        "roots": {...}
    }


# Function to run your YOLO model (you'll need to implement or call your YOLO model here)
from ultralytics import YOLO
from PIL import Image


# Function to run the YOLOv8 model
def run_yolo_model(image_path):
    model = YOLO("C:\\Users\\User\PycharmProjects\TreeTailsModel\YOLO_model\\tree_features\yolov8_training15\weights\\best.pt")
    img = Image.open(image_path)
    results = model(img)

    detections = []
    if results:
        result = results[0]
        boxes = result.boxes

        for box in boxes:
            x, y, width, height = box.xywh[0].tolist()
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            label = result.names[cls]

            detection = {
                'label': label,
                'confidence': conf,
                'bbox': [  # Include 'bbox' key
                    x - width / 2,  # x_min
                    y - height / 2,  # y_min
                    x + width / 2,  # x_max
                    y + height / 2   # y_max
                ]
            }
            detections.append(detection)

    return detections


if __name__ == '__main__':
    app.run(debug=True)
