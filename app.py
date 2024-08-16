from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np

app = Flask(__name__)

# Load YOLO model
model = YOLO("y8best.pt")

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    # Read the image from the request
    file = request.files['image'].read()
    np_img = np.frombuffer(file, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # Make prediction using YOLO model
    results = model.predict(source=img)

    # Convert results to a dictionary (example format)
    output = []
    for result in results:
        output.append({
            'class': int(result.class_id),
            'confidence': float(result.confidence),
            'box': result.box.tolist()
        })

    # Optionally, you could save or return the image with bounding boxes
    # Annotated image with bounding boxes
    annotated_img = results[0].plot()
    _, img_encoded = cv2.imencode('.jpg', annotated_img)
    img_bytes = img_encoded.tobytes()

    return jsonify({'predictions': output}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
