from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import os

app = Flask(__name__, static_folder="../frontend/build", static_url_path="")
CORS(app)  # Allow cross-origin requests

@app.route("/api/analyze", methods=["POST"])
def analyze():
    # Dummy personality analysis response
    data = request.json
    return jsonify({"message": f"Hello, {data.get('name', 'User')}! Your tree drawing has been analyzed."})

# Serve React frontend
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve(path):
    if path != "" and os.path.exists(app.static_folder + "/" + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, "index.html")

if __name__ == "__main__":
    app.run(debug=True)
