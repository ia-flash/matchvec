import json
import cv2
import numpy as np
from flask import Flask, send_from_directory, request
from flask_cors import CORS
from process import predict_class, predict_objects
from urllib.request import urlopen


app = Flask(__name__)
cors = CORS(app)


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


@app.route('/api/object_detection', methods=['POST'])
def api_object_detection():
    images = request.files.getlist('image', None)
    url = request.form.get('url', None)
    res = list()
    if url:
        try:
            resp = urlopen(url)
            img = np.asarray(bytearray(resp.read()), dtype="uint8")
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            res.append(predict_objects(img))
        except Exception as e:
            print(url)
            print(e)
    if images:
        for i in range(len(images)):
            nparr = np.frombuffer(images[i].read(), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            res.append(predict_objects(img))
    return json.dumps(res)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    images = request.files.getlist('image')
    url = request.form.get('url', None)
    res = list()
    if url:
        try:
            resp = urlopen(url)
            img = np.asarray(bytearray(resp.read()), dtype="uint8")
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            res.append(predict_class(img))
        except Exception as e:
            print(url)
            print(e)
    if images:
        for i in range(len(images)):
            nparr = np.frombuffer(images[i].read(), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            res.append(predict_class(img))
    return json.dumps(res)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
