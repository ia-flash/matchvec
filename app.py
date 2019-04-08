from flask import Flask, render_template, Response, render_template_string, send_from_directory, request
from flask_cors import CORS, cross_origin
import json
import os
import  cv2
from PIL import Image
import numpy as np
import mmcv
from mmcv.runner import load_checkpoint

import torch
import torch.nn as nn

from mmdet.models import build_detector
from mmdet.apis import inference_detector
from mmdet.core import get_classes
from mmcv.visualization.color import color_val

import torchvision.models as models
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

app = Flask(__name__)
cors = CORS(app)

device = 0
torch.cuda.set_device(device)

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

# Get label
filename = os.path.join('/model/resnet18-100', 'idx_to_class.json')
with open(filename) as json_data:
    all_categories = json.load(json_data)

checkpoint = torch.load('/model/resnet18-100/model_best.pth.tar')
state_dict =checkpoint['state_dict']

# load multi distributed model
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove 'module.' of dataparallel
    new_state_dict[name]=v

classification_model = models.__dict__['resnet18'](pretrained=True)
classification_model.fc = nn.Linear(512, 99)

classification_model.load_state_dict(new_state_dict)

torch.cuda.set_device(device)

classification_model.cuda(device)
classification_model.eval()


class Crop(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __call__(self, params):
        sample, coords = params
        sample = sample.crop(coords)#[coords[1]: coords[3],
                      #coords[0]: coords[2]]
        return sample

crop = Crop()

def filter_prediction(result, width, height):
    # Detect box of intereset
    result = result[2] # Get class car
    result = result[result[:,4] > 0.4] # Filter by score

    # Plot all boxes and select the closes to the center
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
    return select_box

@app.route('/')
def status():
    return json.dumps({'status': 'ok'})

@app.route('/idx_to_class')
def get_class():

    return json.dumps(idx_to_class)

@app.route('/upload')
def upload_file():
   return render_template('upload.html')

@app.route('/api/object_detection',methods=['POST'])
def api_object_detection():
    # recieve image in files
    image = request.files.get('image', None)
    if image:
        # decode istringmage
        nparr = np.fromstring(image.read(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        height, width = img.shape[:-1]

        # detect boxes
        result = inference_detector(detection_model, img, cfg, device='cuda:%s'%device)

        # Parse resulsts
        res = list()
        for i, class_box in enumerate(result):
            class_box = class_box[class_box[:,4] > 0.4] # Filter by score
            for box in class_box:
                x1, y1 = (int(box[0]), int(box[1]))
                x2, y2 = (int(box[2]), int(box[3]))
                res.append({'bbox': [x1, y1, x2-x1, y2-y1], 'class': CLASS_NAMES[i], 'prob': float(box[4])})

        return json.dumps(res)
    else:
        return json.dumps({'status': 'no image'})

# curl -F image=@test.jpg http://localhost:5000/object_detection
@app.route('/object_detection',methods=['POST'])
def object_detection():
    # recieve image in files
    image = request.files.get('image', None)
    if image:
        # decode istringmage
        nparr = np.fromstring(image.read(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        height, width = img.shape[:-1]

        # detect boxes
        result = inference_detector(detection_model, img, cfg, device='cuda:%s'%device)

        # Parse results
        for i, class_box in enumerate(result):
            class_box = class_box[class_box[:,4] > 0.4] # Filter by score
            for box in class_box:
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
        return Response(img_encoded.tobytes(), mimetype='image/jpeg')
    else:
        return json.dumps({"sta": 'ok'})

@app.route('/api/predict',methods=['POST'])
def api_predict():

    # decode image
    image = request.files["image"]
    nparr = np.fromstring(image.read(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    height, width = img.shape[:-1]

    # Detect all boxes with retina100
    result = inference_detector(detection_model, img, cfg, device='cuda:%s'%device)

    select_box = filter_prediction(result, width, height)

    # Selected box
    if select_box is not None:
        box = select_box
        x1, y1 = (int(box[0]), int(box[1]))
        x2, y2 = (int(box[2]), int(box[3]))
        # Crop and resize
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        preprocess = transforms.Compose([
            crop,
            transforms.Resize([224,224]),
            transforms.ToTensor(),
            normalize,
        ])

        args = (Image.fromarray(img), (x1,y1,x2,y2))
        sample = preprocess(args)

        sample = sample.unsqueeze(0)

        sample = sample.cuda(device)

        # Inference classification model
        output = classification_model(sample)

        softmax = nn.Softmax()
        norm_output = softmax(output)

        # Get max probability
        probs, preds = norm_output.topk(5, 1, True, True)
        pred = preds.data.cpu().tolist()[0]
        prob = probs.data.cpu().tolist()[0]
        return json.dumps({'boxes': [{'bbox': [x1, y1, x2-x1, y2-y1], 'class': all_categories[str(pred[0])], 'prob': float(box[4])}], 'prediction': [{'class': all_categories[str(x)], 'prob': y} for x,y in zip(pred[:5], prob[:5])]})

    else:
        return json.dumps({'status': 'no box'})

# curl -F image=@test.jpg http://localhost:5000/predict
@app.route('/predict',methods=['POST'])
def predict():

    # decode image
    image = request.files["image"]
    nparr = np.fromstring(image.read(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    height, width = img.shape[:-1]

    # Detect all boxes with retina100
    result = inference_detector(detection_model, img, cfg, device='cuda:%s'%device)


    select_box = filter_prediction(result, width, height)

    result = result[2] # Get class car
    result = result[result[:,4] > 0.4] # Filter by score

    for box in result:
        x1, y1 = (int(box[0]), int(box[1]))
        x2, y2 = (int(box[2]), int(box[3]))
        cv2.rectangle(img, (x1, y1), (x2,y2), color_val('blue'), 2)
        cv2.putText(img, "{}, {:.2f}".format(CLASS_NAMES[2], box[4]),
                (int(x1 + 0.005*width), int(y1+0.03*height)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color_val('blue'),
                2)

    # Selected box
    if select_box is not None:
        box = select_box
        x1, y1 = (int(box[0]), int(box[1]))
        x2, y2 = (int(box[2]), int(box[3]))
        # Crop and resize
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        preprocess = transforms.Compose([
            crop,
            transforms.Resize([224,224]),
            transforms.ToTensor(),
            normalize,
        ])

        args = (Image.fromarray(img), (x1,y1,x2,y2))
        sample = preprocess(args)
        sample = sample.unsqueeze(0)

        sample = sample.cuda(device)

        # Inference classification model
        output = classification_model(sample)
        softmax = nn.Softmax()
        norm_output = softmax(output)

        # Get max probability
        probs, preds = norm_output.topk(5, 1, True, True)
        pred = preds.data.cpu().tolist()[0]
        prob = probs.data.cpu().tolist()[0]

        cv2.rectangle(img, (x1, y1), (x2,y2), color_val('green'), 2)

        #Creates two subplots and unpacks the output array immediately
        fig = plt.figure()
        ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3, rowspan=2)
        ax1.imshow(img)
        ax1.axis('off')
        ax2 = plt.subplot2grid((3, 3), (2, 1))
        ax2.barh(range(len(pred[:5])), prob[:5], color='blue')
        ax2.set_yticks(range(len(pred[:5])))
        ax2.set_yticklabels([all_categories[str(x)] for x in pred[:5]])
        ax2.invert_yaxis()  # labels read top-to-bottom
        fig.tight_layout(pad=0)
        fig.canvas.draw()

        # Now we can save it to a numpy array.
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        _, img_encoded = cv2.imencode('.jpg', data)

        return Response(img_encoded.tobytes(), mimetype='image/jpeg')
    else:
        _, img_encoded = cv2.imencode('.jpg', img)
        return Response(img_encoded.tobytes(), mimetype='image/jpeg')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
