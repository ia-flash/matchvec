import os
import json
import cv2
import logging
import requests
from PIL import Image
from itertools import combinations, product
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from utils import timeit
from yolo_detection import Detector
# from ssd_detection import Detector
detector = Detector()

level = logging.DEBUG
logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
        )
logger = logging.getLogger(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


DETECTION_IOU_THRESHOLD = 0.9
DETECTION_SIZE_THRESHOLD = 0.01

# Get label
filename = os.path.join('/model/resnet18-100', 'idx_to_class.json')
with open(filename) as json_data:
    all_categories = json.load(json_data)

checkpoint = torch.load('/model/resnet18-100/model_best.pth.tar', map_location='cpu')
state_dict = checkpoint['state_dict']

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


def filter_by_size(df, image):
    """Filter box too small"""
    height, width = image.shape[:-1]
    surf = width * height
    df = df.assign(
            mean_x=lambda x: x[['x1', 'x2']].mean(axis=1),
            mean_y=lambda x: x[['y1', 'y2']].mean(axis=1),
            dist=lambda x: (
                ((width/2 - x['mean_x']) ** 2 +
                    (height/2 - x['mean_y'])**2).pow(1./2)),
            surf_box=lambda x: (x['x2'] - x['x1']) * (x['y2'] - x['y1']),
            surf_ratio=lambda x: x['surf_box'] / surf
            )
    df = df[(df['surf_ratio'] > DETECTION_SIZE_THRESHOLD)]
    return df


def filter_by_iou(df):
    """Filter box of car and truck when IoU>DETECTION_IOU_THRESHOLD """
    df['surf_box'] = (df['x2'] - df['x1']) * (df['y2'] - df['y1'])
    df_class = df[df['class_name'].isin(['car', 'truck'])].groupby('class_name')
    prod_class = combinations(df_class, 2)
    id_to_drop = []
    for (class_a, df_a), (class_b, df_b) in prod_class:
        for (id1, vec1), (id2, vec2) in product(df_a.iterrows(), df_b.iterrows()):
            iou = IoU(vec1, vec2)
            if iou > DETECTION_IOU_THRESHOLD:
                # print('drop truck')
                if class_a == 'truck':
                    id_to_drop += [id1]
                elif class_b == 'truck':
                    id_to_drop += [id2]
    # drop trucks overlapping car
    df = df.drop(id_to_drop)
    return df


def test_app():
    # url = 'http://matchvec:5000/api/object_detection'
    url = 'http://matchvec:5000/api/predict'
    files = {'image': open('clio-peugeot.jpg', 'rb')}
    res = requests.post(url, files=files)
    logger.debug(res.text)


def test_app_multiple():
    url = 'http://matchvec:5000/api/object_detection'
    files = [('image', open('clio-peugeot.jpg', 'rb')), ('image', open('cliomegane.jpg', 'rb'))]
    res = requests.post(url, files=files)
    logger.debug(res.text)


@timeit
def predict_objects(img):
    result = detector.prediction(img)
    df = detector.create_df(result, img)

    df = filter_by_size(df)

    df = filter_by_iou(df)

    cols = ['x1', 'y1', 'x2', 'y2', 'class_name', 'confidence', 'label']
    return df[cols].to_dict(orient='records')


@timeit
def predict_class(img):
    result = detector.prediction(img)
    df = detector.create_df(result, img)

    # Filter by class
    df = df[df['class_name'] == 'car']

    selected_boxes = list(
            zip(
                [Image.fromarray(img)]*len(df),
                df[['x1', 'y1', 'x2', 'y2']].values.tolist()
                )
            )

    # Selected box
    if len(selected_boxes) > 0:
        # Crop and resize
        crop = Crop()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        preprocess = transforms.Compose([
            crop,
            transforms.Resize([224, 224]),
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
                    pred=[[all_categories[str(x)] for x in pred[i]] for i in range(len(pred))],
                    prob=prob,
                    label=lambda x: (
                        x['pred'].apply(lambda x: x[0]) +
                        ": " + (
                            x['prob'].apply(lambda x: x[0])
                            .astype(str).str.slice(stop=4)
                            )
                        )
                    )
            cols = ['x1', 'y1', 'x2', 'y2', 'pred', 'prob', 'class_name',
                    'confidence', 'label']
            return df[cols].to_dict(orient='records')
    else:
        return list()


if __name__ == '__main__':
    img = cv2.imread('clio-punto-megane.jpg')
    #img = cv2.imread('image.jpg')
    print(img.shape)
    #res = predict_objects(img)
    res = predict_class(img)
    print(res)
    #test_app()
    #test_app_multiple()
