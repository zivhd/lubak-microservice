from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np

app = Flask(__name__)

model = YOLO("y8best.pt")

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image'].read()
    np_img = np.frombuffer(file, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    results = model.predict(source=img)

    output = []
    for result in results:
        output.append({
            'class': int(result.class_id),
            'confidence': float(result.confidence),
            'box': result.box.tolist()
        })


    return jsonify({'predictions': output}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
