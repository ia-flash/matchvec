import cv2
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from collections import OrderedDict
from typing import List, Tuple, Dict
from matchvec.utils import timeit, logger

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


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
    def __init__(self, classification_model):
        """TODO: to be defined1. """
        # Get label
        print('****** TOTO ********')
        filename = os.path.join(os.environ['BASE_MODEL_PATH'], classification_model,  'idx_to_class.json')
        with open(filename) as json_data:
            self.all_categories = json.load(json_data)
            class_number = len(self.all_categories)

        checkpoint = torch.load(
                os.path.join(os.environ['BASE_MODEL_PATH'], classification_model, 'model_best.pth.tar'),
                map_location='cpu'
                )

        state_dict = checkpoint['state_dict']

        new_state_dict: Dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove 'module.' of dataparallel
            new_state_dict[name] = v

        self.classification_model = models.__dict__['resnet18'](pretrained=True)
        self.classification_model.fc = nn.Linear(512, class_number)
        self.classification_model.load_state_dict(new_state_dict)
        print(type(device))
        if device.type == 'cuda' :
            torch.cuda.set_device(device)
            self.classification_model.cuda(device)

        self.classification_model.eval()


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
            if device.type == 'cuda':
                inp = inp.cuda(device, non_blocking=True)

            output = self.classification_model(inp)

            softmax = nn.Softmax(dim=1)
            norm_output = softmax(output)

            probs, preds = norm_output.topk(5, 1, True, True)
            pred = preds.data.cpu().tolist()
            pred_class = [
                    [self.all_categories[str(x)] for x in pred[i]]
                    for i in range(len(pred))
                    ]
            prob = probs.data.cpu().tolist()
            final_pred.extend(pred_class)
            final_prob.extend(prob)
        return final_pred, final_prob


def detect_faces(image, ind):
    THRESHOLD = 0.2

    # load our serialized model from disk
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(
            '/model/face_detection/deploy.prototxt.txt',
            '/model/face_detection/res10_300x300_ssd_iter_140000.caffemodel')

    # load the input image and construct an input blob for the image
    # by resizing to a fixed 300x300 pixels and then normalizing it
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the detections and
    # predictions
    print("[INFO] computing object detections...")
    net.setInput(blob)
    detections = net.forward()
    count = 0

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > THRESHOLD:
                count += 1
                # compute the (x, y)-coordinates of the bounding box for the
                # object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # draw the bounding box of the face along with the associated
                # probability
                text = "{:.2f}%".format(confidence * 100) + 'Count ' + str(count)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
                cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    cv2.imwrite('image{}.jpg'.format(ind), image)
    logger.debug('Count: {}'.format(count))
