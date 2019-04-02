from flask import Flask, render_template, Response, render_template_string, send_from_directory, request
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
import torch.multiprocessing as mp

import torchvision.models as models
import torch.distributed as dist
from torchvision.transforms.functional import to_tensor, normalize, resize
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn


app = Flask(__name__)

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
    nparr = np.fromstring(image.read(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    height, width = img.shape[:-1]
    print(height, width)
    # detect boxes
    result = inference_detector(detection_model, img, cfg, device='cuda:%s'%device)
    for i, class_box in enumerate(result):
        print('The %s th box'%i)
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
        cv2.rectangle(img, (x1, y1), (x2,y2), color_val('blue'), 2)
        cv2.putText(img, "{}, {:.2f}".format(CLASS_NAMES[2], box[4]),
                (int(x1 + 0.005*width), int(y1+0.03*height)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color_val('blue'),
                2)

    # Selected box
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
    
    #im = cv2.resize(img[y1:y2, x1:x2],(224, 224))

    # To tensor, normalize and reshape to match input
    #im = to_tensor(im)
    #sample = normalize(tensor=im, mean=[0.485, 0.456, 0.406],
    #        std=[0.229, 0.224, 0.225])

    #print(sample.size())

    sample = sample.unsqueeze(0)

    sample = sample.cuda(device)
    #print(im.data)

    # Inference classification model
    output = classification_model(sample)


    print(output.size())

    softmax = nn.Softmax()
    norm_output = softmax(output)
    print(norm_output.data)

    # Get max probability
    probs, preds = norm_output.topk(5, 1, True, True)
    print(probs.data)
    pred = preds.data.cpu().tolist()

    result = ""
    if len(pred) > 0:
        first = all_categories[str(pred[0][0])]
        first_prob = probs.data.cpu().tolist()[0][0]
        second = all_categories[str(pred[0][1])]
        second_prob = probs.data.cpu().tolist()[0][1]
        third = all_categories[str(pred[0][2])]
        third_prob = probs.data.cpu().tolist()[0][2]

    cv2.rectangle(img, (x1, y1), (x2,y2), color_val('green'), 2)
    cv2.putText(img, "{}: {:.3f}; {}: {:.3f}; {}: {:.3f}".format(
        first, first_prob, second, second_prob, third, third_prob),
            (int(x1 + 0.005*width), int(y2-0.03*height)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color_val('green'),
            2)
    _, img_encoded = cv2.imencode('.jpg', img)

    return Response(img_encoded.tobytes(), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
