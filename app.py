from flask import Flask, render_template, Response, render_template_string, send_from_directory, request
import json
import os
import cv2
import numpy as np
import mmcv
from mmcv.runner import load_checkpoint

from mmdet.models import build_detector
from mmdet.apis import inference_detector
from mmdet.core import get_classes
from mmcv.visualization.color import color_val


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

COLORS = np.random.uniform(0, 255, size=(100, 3))
CLASS_NAMES = get_classes('coco')

@app.route('/')
def status():
    return json.dumps({'status': 'ok'})

@app.route('/idx_to_class')
def get_class():

    return json.dumps(idx_to_class)

@app.route('/upload')
def upload_file():
   return render_template('upload.html')

# curl -F image=@test.jpg http://localhost:5000/object_detection
@app.route('/object_detection',methods=['POST'])
def object_detection():
    # decode image
    image = request.files["image"]
    #print(image)
    nparr = np.fromstring(image.read(), np.uint8)

    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    height, width = img.shape[:-1]

    result = inference_detector(detection_model, img, cfg, device='cuda:%s'%device)

    for i, class_box in enumerate(result):
        for box in class_box:
            if box[4] > 0.4:
                x1, y1 = (int(box[0]), int(box[1]))
                x2, y2 = (int(box[2]), int(box[3]))
                cv2.rectangle(img, (x1, y1), (x2,y2), COLORS[i], 2)
                cv2.putText(img, "{}, {:.2f}".format(CLASS_NAMES[i], box[4]),
                        (int(x1 + 0.005*width), int(y1+0.03*height)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        COLORS[i],
                        2)
    _, img_encoded = cv2.imencode('.jpg', img)
    send_image = img_encoded.tobytes()

    return Response(img_encoded.tobytes(), mimetype='image/jpeg')

# curl -F image=@test.jpg http://localhost:5000/predict
@app.route('/predict',methods=['POST'])
def predict():

    # decode image
    image = request.files["image"]
    #print(image)
    nparr = np.fromstring(image.read(), np.uint8)

    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    height, width = img.shape[:-1]
    # TODO: Check si l'image est bien en rgb..

    # TODO: detect box of interest
    result = inference_detector(detection_model, img, cfg, device='cuda:%s'%device)
    # TODO: choose box of interest and return its coords

    result = result[2] # Get class car
    result = result[result[:,4] > 0.4] # Filter by score

    # TODO: crop

    # TODO: predit

    marque = 'RENAULT'
    modele = 'CLIO'
    dist_box = 99999
    select_box = None
    for box in result:
        x1, y1 = (int(box[0]), int(box[1]))
        x2, y2 = (int(box[2]), int(box[3]))

        mean_x = (x1+x2)/2
        mean_y = (y1+y2)/2
        surf_box = (x2 - x1)*(y2-y1)
        dist = (width/2 - mean_x) + (height/2 - mean_y)
        surf_ratio = surf_box / (width * height)
        if dist < dist_box:
            if surf_ratio > 0.2:
                select_box = box
                dist_box = dist
        cv2.rectangle(img, (x1, y1), (x2,y2), color_val('blue'), 2)
        cv2.putText(img, "{}, {:.2f}".format(CLASS_NAMES[2], box[4]),
                (int(x1 + 0.005*width), int(y1+0.03*height)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color_val('blue'),
                2)

    box = select_box
    x1, y1 = (int(box[0]), int(box[1]))
    x2, y2 = (int(box[2]), int(box[3]))
    cv2.rectangle(img, (x1, y1), (x2,y2), color_val('green'), 2)
    cv2.putText(img, "{}, {:.2f}".format(CLASS_NAMES[2], box[4]),
            (int(x1 + 0.005*width), int(y1+0.03*height)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color_val('green'),
            2)
    _, img_encoded = cv2.imencode('.jpg', img)
    send_image = img_encoded.tobytes()

    #return 
    #json.dumps(dict(marque=marque,modele=modele,shape=img.shape,image=img_encoded))

    #print(img_encoded.tobytes())
    #return render_template('result.html', **dict(marque=marque,modele=modele,shape=img.shape,image=send_image))
    return Response(img_encoded.tobytes(), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')
