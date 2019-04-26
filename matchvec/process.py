import os
import json
import cv2
import logging
import pandas as pd
from PIL import Image
from itertools import combinations
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from utils import timeit

level = logging.DEBUG
logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
        )
logger = logging.getLogger(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# DETECTION OPENCV
# DETECTION_MODEL = 'faster_rcnn_resnet101_coco_2018_01_28/'
DETECTION_MODEL = 'ssd_mobilenet_v2_coco_2018_03_29/'
DETECTION_THRESHOLD = 0.4
cvNet = cv2.dnn.readNetFromTensorflow(
        os.path.join('/model', DETECTION_MODEL, 'frozen_inference_graph.pb'),
        os.path.join('/model', DETECTION_MODEL, 'config.pbtxt'))

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

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class DatasetList(torch.utils.data.Dataset):
    def __init__(self, samples, transform=None, loader=default_loader, target_transform=None):
        self.samples = samples
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in dataframe"))

        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        image, coords = self.samples[index] # coords [x1,y1,x2,y2]
        args = (image, coords)

        if self.transform is not None:
            sample = self.transform(args)
        return sample

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        return "Dataset size {}".format(self.__len__())


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


def IoU(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA['x1'], boxB['x1'])
    yA = max(boxA['y1'], boxB['y1'])
    xB = min(boxA['x2'], boxB['x2'])
    yB = min(boxA['y2'], boxB['y2'])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxA['surf_box'] + boxB['surf_box'] - interArea)

    # return the intersection over union value
    return iou

def create_df(result, width, height):
    df = pd.DataFrame(result, columns=['_', 'class_id', 'confidence', 'x1', 'y1', 'x2', 'y2'])
    df = df.assign(
            x1 = (df['x1'] * width).astype(int).clip(0),
            y1 = (df['y1'] * height).astype(int).clip(0),
            x2 = (df['x2'] * width).astype(int),
            y2 = (df['y2'] * height).astype(int),
            class_name = df['class_id'].apply(lambda x: CLASS_NAMES[str(int(x))])
            )
    return df


@timeit
def predict_objects(img):
    height, width = img.shape[:-1]
    surf = width * height

    cvNet.setInput(cv2.dnn.blobFromImage(img, size=(300, 300), swapRB=False, crop=False))
    cvOut = cvNet.forward()
    result = cvOut[0,0,:,:]

    df = create_df(result, width, height)

    # Filter
    df = df[df['confidence'] > DETECTION_THRESHOLD] #filter by score

    df['label'] = df['class_name'] + ': ' + df['confidence'].astype(str).str.slice(stop=4)
    cols = ['x1', 'y1', 'x2', 'y2', 'class_name', 'confidence', 'label']
    return json.dumps(df[cols].to_dict(orient='records'))

@timeit
def predict_class(img):
    # Make predictions
    height, width = img.shape[:-1]
    cvNet.setInput(cv2.dnn.blobFromImage(img, size=(300, 300), swapRB=False, crop=False))
    cvOut = cvNet.forward()
    result = cvOut[0,0,:,:]

    df = create_df(result, width, height)

    # Filter
    df = df[df['confidence'] > DETECTION_THRESHOLD] # filter by score
    df = df[df['class_name'] == 'car'] # filter by class

    selected_boxes = list(zip([Image.fromarray(img)]*len(df),df[['x1','y1','x2','y2']].values.tolist()))

    # Selected box
    if len(selected_boxes) > 0:
        # Crop and resize
        crop = Crop()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        preprocess = transforms.Compose([
            crop,
            transforms.Resize([224,224]),
            transforms.ToTensor(),
            normalize,
        ])

        val_loader = torch.utils.data.DataLoader(
                DatasetList(selected_boxes, transform=preprocess),
                batch_size=256, shuffle=False)

        for inp in val_loader:
            output = classification_model(inp)

            softmax = nn.Softmax()
            norm_output = softmax(output)

            probs, preds = norm_output.topk(5, 1, True, True)
            pred = preds.data.cpu().tolist()
            prob = probs.data.cpu().tolist()

            df = df.assign(
                    pred = [[all_categories[str(x)] for x in pred[i]] for i in range(len(pred))],
                    prob = prob,
                    )
            df['label'] = df['pred'].apply(lambda x: x[0]) + ": " + df['prob'].apply(lambda x: x[0]).astype(str).str.slice(stop=4)
            logger.debug("After all")
            logger.debug("\n {}".format(df.values))
            cols = ['x1', 'y1', 'x2', 'y2', 'pred', 'prob', 'class_name', 'confidence', 'label']
            return json.dumps(df[cols].to_dict(orient='records'))
    else:
        return json.dumps(list())

if __name__ == '__main__':
    img = cv2.imread('clio-punto-megane.jpg')
    #res = predict_objects(img)
    res = predict_class(img)
    print(res)
