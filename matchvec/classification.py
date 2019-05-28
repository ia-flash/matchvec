"""Classification Marque Modèle"""
import os
import cv2
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from collections import OrderedDict
from typing import List, Tuple, Dict
from utils import timeit

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CLASSIFICATION_MODEL = os.getenv('CLASSIFICATION_MODEL')

# Get label
filename = os.path.join('/model', CLASSIFICATION_MODEL,  'idx_to_class.json')
with open(filename) as json_data:
    all_categories = json.load(json_data)
    CLASS_NUMBER = len(all_categories)

checkpoint = torch.load(
        os.path.join('/model', CLASSIFICATION_MODEL, 'model_best.pth.tar'),
        map_location='cpu'
        )
state_dict = checkpoint['state_dict']

new_state_dict: Dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:]  # remove 'module.' of dataparallel
    new_state_dict[name] = v


def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    #bz, nc, h, w = feature_conv.shape
    nc, h, w = feature_conv.shape
    print("FEAture")
    print(feature_conv.shape)
    #myfeature_conf = np.expand_dims(feature_conv[0], axis=0)
    myfeature_conf = np.expand_dims(feature_conv, axis=0)
    print(myfeature_conf.shape)
    print("weift")
    print(weight_softmax.shape)
    print(weight_softmax[0].shape)
    output_cam = []
    for idx in class_idx:
        #cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = weight_softmax[idx].dot(myfeature_conf.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

#def hook_feature(module, input, output):
#    #features_blobs.append(output.data.cpu().numpy())
#    global features_blobs
#    features_blobs = output.data.cpu().numpy()

class DatasetList(torch.utils.data.Dataset):
    """ Datalist generator

    Args:
        samples: Samples to use for inference
        transform: Transformation to be done to samples
        target_transform: Transformation done to the targets
    """
    def __init__(self, samples: Tuple[np.ndarray, List[float]],
                 transform=None, target_transform=None):
        self.samples = samples
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in dataframe"))

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        image, coords = self.samples[index]  # coords [x1,y1,x2,y2]
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
        params: Tuple containing the sample and coordinates. The image is cropped using the coordiantes.
    """
    def __call__(self, params):
        sample, coords = params
        # [coords[1]: coords[3], coords[0]: coords[2]]
        sample = sample.crop(coords)
        return sample


class Classifier(object):
    """Classifier for marque et modèle

    Classifies images using a pretrained model.
    """

    @timeit
    def __init__(self):
        """TODO: to be defined1. """
        self.classification_model = models.__dict__['resnet18'](pretrained=True)
        self.classification_model.fc = nn.Linear(512, CLASS_NUMBER)
        self.classification_model.load_state_dict(new_state_dict)
        self.classification_model.eval()
        self.features_blobs = None

    def hook_feature(self, module, input, output):
        #features_blobs.append(output.data.cpu().numpy())
        self.features_blobs = output.data.cpu().numpy()


    def create_test_set(self, selected_boxes):
        # Crop and resize
        crop = Crop()
        # Nomalize
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
        return val_loader


    @timeit
    def prediction(self, selected_boxes: Tuple[np.ndarray, List[float]]):
        """Inference in image

        1. Crops, normalize and transforms the image to tensor
        2. The image is forwarded to the resnet model
        3. The results are concatenated

        Args:
            selected_boxes: Contains a List of Tuple with the image and coordinates of the crop.

        Returns:
            (final_pred, final_prob): The result is two lists with the top 5 class prediction and the probabilities
        """
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

        final_pred: List = list()
        final_prob: List = list()
        for inp in val_loader:
            output = self.classification_model(inp)

            softmax = nn.Softmax(dim=1)
            norm_output = softmax(output)

            probs, preds = norm_output.topk(5, 1, True, True)
            pred = preds.data.cpu().tolist()
            pred_class = [
                    [all_categories[str(x)] for x in pred[i]]
                    for i in range(len(pred))
                    ]
            prob = probs.data.cpu().tolist()
            final_pred.extend(pred_class)
            final_prob.extend(prob)
        return final_pred, final_prob

    @timeit
    def generate_CAM(self, selected_boxes, image):
        # Top predictions
        n_pred = 5

        # Last conv layer for Resnet18
        finalconv_name = 'layer4'

        # Hook the feature extractor
        self.classification_model._modules.get(finalconv_name).register_forward_hook(self.hook_feature)

        # get the softmax weight
        params = list(self.classification_model.parameters())
        weight_softmax = np.squeeze(params[-2].data.numpy())

        # Get test set with transformations
        val_loader = self.create_test_set(selected_boxes)

        final_pred: List = list()
        final_prob: List = list()
        final_CAM: List = list()
        for inp in val_loader:
            logit = self.classification_model(inp)

            softmax = nn.Softmax(dim=1)
            h_x = softmax(logit)
            probs, idx = h_x.sort(0, True)
            _, preds = h_x.topk(n_pred, 1, True, True)
            pred = preds.data.cpu().tolist()
            pred_class = [
                    [all_categories[str(x)] for x in pred[i]]
                    for i in range(len(pred))
                    ]
            prob = probs.data.cpu().tolist()
            final_prob.extend(prob)
            final_pred.extend(pred_class)

            for i in range(len(inp)):
                coords = selected_boxes[i][1]
                crop_img = image[coords[1]: coords[3], coords[0]: coords[2]]
                height, width, _ = crop_img.shape
                features_blob = self.features_blobs[i]
                print("Feature blob", features_blob.shape)
                CAMs = returnCAM(features_blob, weight_softmax, pred[i][:n_pred])
                print("Cams", len(CAMs))
                CAM_images = list()
                for ind in range(len(CAMs)):
                    heatmap = cv2.applyColorMap(cv2.resize(CAMs[ind], (width, height)), cv2.COLORMAP_JET)
                    result = heatmap * 0.6 + crop_img * 0.5
                    cv2.putText(result, pred_class[i][ind],
                                (0, int(height/10)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 255, 255), 2)
                    CAM_images.append(result)
                    cv2.imwrite('imgCAM{}-{}.jpg'.format(i, ind), result)
                final_CAM.append(CAM_images)

        return dict(pred=final_pred, prod=final_prob, CAM=final_CAM)

if __name__ == "__main__":
    from PIL import Image
    from ssd_detection import Detector
    detector = Detector()
    classifier = Classifier()

    image = cv2.imread("./tests/clio-peugeot.jpg")
    #height, width, _ = image.shape
    #selected_boxes = list(
    #        zip(
    #            [Image.fromarray(image)],
    #            [[0, 0, int(width), int(height)]]
    #            )
    #        )

    result = detector.prediction(image)
    df = detector.create_df(result, image)
    selected_boxes = list(
            zip(
                [Image.fromarray(image)]*len(df),
                df[['x1', 'y1', 'x2', 'y2']].values.tolist()
                )
            )

    result = classifier.generate_CAM(selected_boxes, image)
    print(result.keys())
    pred, prob = classifier.prediction(selected_boxes)
    print(pred, prob)
