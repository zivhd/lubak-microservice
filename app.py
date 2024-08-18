from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io
import numpy as np
from inference_sdk import InferenceHTTPClient

app = Flask(__name__)



@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    CLIENT = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key="fi2WcHFLbFMDCYM6sThP"
    )
    
    file = request.files['image'].read()
    image = Image.open(io.BytesIO(file)).convert('RGB')
    img = np.array(image)

    result = CLIENT.infer(img, model_id="pothole-detection-yolov8/1")
    
    # Extracting predictions
    predictions = []
    for detection in result['predictions']:
        prediction = {
            'confidence': detection['confidence'],
            'class': detection['class'],
            'class_id': detection['class_id'],
            'detection_id': detection['detection_id']
        }
        predictions.append(prediction)
    print(predictions)
    
    return jsonify({'predictions': predictions}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
