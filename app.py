import os
import json
import cv2
import numpy as np
from PIL import Image
from flask import Flask, render_template, send_from_directory, request
from flask_cors import CORS, cross_origin
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

app = Flask(__name__)
cors = CORS(app)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# DETECTION OPENCV
DETECTION_MODEL = 'faster_rcnn_resnet101_coco_2018_01_28/'
#DETECTION_MODEL = 'ssd_mobilenet_v2_coco_2018_03_29/'
cvNet = cv2.dnn.readNetFromTensorflow(
        os.path.join('/model', DETECTION_MODEL, 'frozen_inference_graph.pb'),
        os.path.join('/model', DETECTION_MODEL, 'config.pbtxt'))

COLORS = np.random.uniform(0, 255, size=(100, 3))
# Class names
filename = os.path.join('/model', DETECTION_MODEL, 'labels.json')
with open(filename) as json_data:
    CLASS_NAMES = json.load(json_data)

# Get label
filename = os.path.join('/model/resnet18-100', 'idx_to_class.json')
with open(filename) as json_data:
    all_categories = json.load(json_data)

checkpoint = torch.load('/model/resnet18-100/model_best.pth.tar', map_location='cpu')
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
    result = result[result[:,2] > 0.4] # Filter by score

    dist_box = 99999
    select_box = None
    for box in result:
        x1 = box[3] * width
        y1 = box[4] * height
        x2 = box[5] * width
        y2 = box[6] * height

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

@app.route('/<path:path>')
def build(path):
    return send_from_directory('dist', path)

@app.route('/')
def status():
    return send_from_directory('dist', "index.html")

@app.route('/preview')
def preview():
    return send_from_directory('dist', "index.html")

@app.route('/sivnorm')
def sivnorm():
    return send_from_directory('dist', "index.html")

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

        cvNet.setInput(cv2.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False))
        cvOut = cvNet.forward()
        result = cvOut[0,0,:,:]
        result = result[result[:,2] > 0.4] # Filter by score

        res = list()
        for i, detection in enumerate(cvOut[0,0,:,:]):
            score = float(detection[2])
            if score > 0.4:
                x1 = max(detection[3] * width, 0)
                y1 = max(detection[4] * height, 0)
                x2 = detection[5] * width
                y2 = detection[6] * height
                res.append({
                    'bbox': [x1, y1, x2-x1, y2-y1],
                    'class': "{}: {:.2f}".format(
                        CLASS_NAMES[str(int(detection[1]))], detection[2]
                        ),
                    'prob': float(detection[2])})

        return json.dumps(res)
    else:
        return json.dumps({'status': 'no image'})

@app.route('/api/predict',methods=['POST'])
def api_predict():

    # decode image
    image = request.files["image"]
    nparr = np.fromstring(image.read(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    height, width = img.shape[:-1]
    cvNet.setInput(cv2.dnn.blobFromImage(img, size=(300, 300), swapRB=True, 
        crop=False))
    cvOut = cvNet.forward()
    result = cvOut[0,0,:,:]

    select_box = filter_prediction(result, width, height)

    # Selected box
    if select_box is not None:
        box = select_box
        x1 = box[3] * width
        y1 = box[4] * height
        x2 = box[5] * width
        y2 = box[6] * height
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
