from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io
import numpy as np

app = Flask(__name__)

model = YOLO("model.pt")

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image'].read()
    image = Image.open(io.BytesIO(file)).convert('RGB')
    img = np.array(image)

    results = model.predict(source=img)

    results_list = {
        "predictions": []
      }
    for result in results:
        objects = []

        class_npy = result.cpu().boxes.cls.numpy()
        i = 0
        temp_class = []
        for classes in class_npy:
          temp_class.append(classes)
          i += 1

        i = 0
        temp_boxes = []
        x_point = []
        y_point = []
        x1 = []
        y1 = []
        x2 = []
        y2 = []
        boxes_npy = result.cpu().boxes.xyxy.numpy()
        for box in boxes_npy:
          x_point.append((box[0]+box[2])/2)
          y_point.append(box[3])
          x1.append(box[0])
          y1.append(box[1])
          x2.append(box[2])
          y2.append(box[3])

        for x in range(len(temp_class)):
          full_boxes = [int(x1[x]), int(y1[x]), int(x2[x]), int(y2[x])]
          c_point = [int(x_point[x]), int(y_point[x])]
          if int(temp_class[x]) == 3:
            objects.append({"class": int(temp_class[x]), "bounding_box": full_boxes, "bottom_center": c_point})
        results_list["predictions"].append({"objects": objects})

    return jsonify({'predictions': results_list}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
