import os
import json
import cv2
import logging
import numpy as np
#import pandas as pd
from flask import Flask, render_template, send_from_directory, request
from flask_cors import CORS
from process import predict_class, predict_objects

app = Flask(__name__)
cors = CORS(app)

level = logging.DEBUG
logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
logger = logging.getLogger(__name__)

@app.route('/<path:path>')
def build(path):
    return send_from_directory('../dist', path)

@app.route('/')
def status():
    return send_from_directory('../dist', "index.html")

@app.route('/preview')
def preview():
    return send_from_directory('../dist', "index.html")

@app.route('/sivnorm')
def sivnorm():
    return send_from_directory('../dist', "index.html")

@app.route('/api/object_detection',methods=['POST'])
def api_object_detection():
    # recieve image in files
    image = request.files.get('image', None)
    if image:
        # decode istringmage
        nparr = np.fromstring(image.read(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return predict_objects(img)
    else:
        return json.dumps({'status': 'no image'})

@app.route('/api/predict',methods=['POST'])
def api_predict():
    # decode image
    image = request.files["image"]
    if image:
        nparr = np.fromstring(image.read(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return predict_class(img)
    else:
        return json.dumps({'status': 'no image'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
