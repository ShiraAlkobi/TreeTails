from flask import Flask, jsonify
from app import run_yolo_model  # import your YOLO model processing function

app = Flask(__name__)

# Image path (you can change this to your image's actual path)
IMAGE_PATH = r"C:\Users\User\Desktop\tree.jpg"


@app.route('/process_image', methods=['GET'])
def process_image():
    # Run YOLO model and get results
    detections = run_yolo_model(IMAGE_PATH)

    # Return the results as JSON or display them
    return jsonify({"detections": detections})


if __name__ == '__main__':
    app.run(debug=True)
