from flask import Flask, render_template, Response, render_template_string, send_from_directory, request
import json
import os
import cv2
import numpy as np
import io
import mmcv
from mmcv.runner import load_checkpoint

from mmdet.models import build_detector
from mmdet.apis import inference_detector

app = Flask(__name__)

device = 0

filename = os.path.join(os.path.join('/model','resnet18-100', 'idx_to_class.json'))
with open(filename) as json_data:
   idx_to_class = json.load(json_data)

# DETECTION
modele = dict(conf="retinanet_x101_64x4d_fpn_1x",
          checkpoint="retinanet_x101_64x4d_fpn_1x_20181218-2f6f778b")

cfg = mmcv.Config.fromfile('/usr/src/app/configs/%s.py'%modele['conf'])
cfg.model.pretrained = None

detection_model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
#model.to(device)# = MMDataParallel(model, device_ids=range(num_gpu))
_ = load_checkpoint(detection_model, '/model/%s.pth'%modele['checkpoint'])

@app.route('/')
def status():
    return json.dumps({'status': 'ok'})

@app.route('/idx_to_class')
def get_class():

    return json.dumps(idx_to_class)

@app.route('/upload')
def upload_file():
   return render_template('upload.html')

# curl -F image=@test.jpg http://localhost:5000/predict
@app.route('/predict',methods=['POST'])
def predict():

    # decode image
    image = request.files["image"]
    print(image)
    nparr = np.fromstring(image.read(), np.uint8)

    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    print(img.shape)
    # TODO: Check si l'image est bien en rgb..


    # TODO: detect box of interest
    result = inference_detector(detection_model, img, cfg, device='cuda:%s'%device)
    print(result)
    # TODO: choose box of interest and return its coords

    # TODO: crop

    # TODO: predit

    marque = 'RENAULT'
    modele = 'CLIO'
    return json.dumps(dict(marque=marque,modele=modele,shape=img.shape))

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')
