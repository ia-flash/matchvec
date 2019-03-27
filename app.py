from flask import Flask, render_template, Response, render_template_string, send_from_directory, request
import json
import os
import cv2
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

class Args():
   def __init__( self):
      self.arch = 'resnet18'
      self.batch_size = 256
      self.dist_url = 'tcp://127.0.0.1:1235'
      self.dist_backend = 'nccl'
      self.num_classes = 99
      self.world_size = 1
      self.rank = 0
      self.gpu = 0
      self.resume = '/model/resnet18-100/model_best.pth.tar'
      self.data = '/model/resnet18-100'
      self.pretrained = False
      self.workers = 4
      self.multiprocessing_distributed = False

def load_model():
    args = Args()

    # Count gpus
    ngpus_per_node = torch.cuda.device_count()

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    # finetune model
    model.fc = nn.Linear(512, args.num_classes)

    args.rank = args.rank * ngpus_per_node + args.gpu
    # This blocks until all processes have joined.
    dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank)

    # set gpu
    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)
    # When using a single GPU per process and per
    # DistributedDataParallel, we need to divide the batch size
    # ourselves based on the total number of GPUs we have
    args.batch_size = int(args.batch_size / ngpus_per_node)
    args.workers = int(args.workers / ngpus_per_node)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # cudnn auto-tuner to find the best algorithm to use for your hardware.
    cudnn.benchmark = True

    return model


classification_model = load_model()

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

    # detect boxes
    result = inference_detector(detection_model, img, cfg, device='cuda:%s'%device)

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
    im = cv2.resize(img[y1:y2, x1:x2],(224, 224))

    # To tensor, normalize and reshape to match input
    im = to_tensor(im)
    im = normalize(tensor=im, mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]).reshape(1,3,224,224)
    im = im.cuda(0, non_blocking=True)

    # Inference classification model
    output = classification_model(im)

    # Get label
    filename = os.path.join('/model/resnet18-100', 'idx_to_class.json')
    with open(filename) as json_data:
        all_categories = json.load(json_data)

    norm_output = output.add(-output.min()).div(output.add(-output.min()).sum()).mul(100)

    # Get max probability
    probs, preds = norm_output.topk(5, 1, True, True)
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
