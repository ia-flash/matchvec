"""Classification Marque Modèle"""
import cv2
import io
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from collections import OrderedDict
from typing import List, Tuple, Dict
from utils import timeit, logger
import onnxruntime
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

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


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def detect_faces(image, ind):
    THRESHOLD = 0.2

    # load our serialized model from disk
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(
            '/model/face_detection/deploy.prototxt.txt',
            '/model/face_detection/res10_300x300_ssd_iter_140000.caffemodel')

    # load the input image and construct an input blob for the image
    # by resizing to a fixed 300x300 pixels and then normalizing it
    #image = cv2.imread('/app/radars/00202_20180118_223914_00030_1.jpg')
    #image = cv2.imread('/app' + '/radars/00202_20180118_223914_00030_1.jpg')
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
        print(type(device))
        if device.type == 'cuda' :
            torch.cuda.set_device(device)
            self.classification_model.cuda(device)

        self.classification_model.eval()


    def export_model(self):
        """ Export model"""
        # Input to the model
        batch_size = 1
        x = torch.randn(batch_size, 3, 224, 224, requires_grad=True)
        # Export the model
        torch_out = torch.onnx._export(self.classification_model,             # model being run
                                       x,                       # model input (or a tuple for multiple inputs)
                                       "classifcation_model.onnx", # where to save the model (can be a file or file-like object)
                                       export_params=True)      # store the trained parameter weights inside the model file


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
            if device.type == 'cuda' :
                inp = inp.cuda(device, non_blocking=True)

            #logger.debug(inp.shape)
            output = self.classification_model(inp)
            logger.debug(inp.data.numpy()[0])
            #logger.debug(inp.data.numpy().shape)
            #res = self.session.run([self.output_name], {self.input_name: inp.data.numpy()})
            #logger.debug('torch')
            #logger.debug(output.shape)
            #logger.debug('onnx')
            #logger.debug(res[0].shape)
            #logger.debug('pred')
            #pred_onnx = np.argmax(res[0], axis=1)
            #prob_onnx = np.max(res[0], axis=1)
            #logger.debug([[i] for i in pred_onnx])
            #logger.debug([[i] for i in prob_onnx])


            softmax = nn.Softmax(dim=1)
            norm_output = softmax(output)
            logger.debug('norm output')
            logger.debug(norm_output.shape)

            probs, preds = norm_output.topk(5, 1, True, True)
            pred = preds.data.cpu().tolist()
            pred_class = [
                    [all_categories[str(x)] for x in pred[i]]
                    for i in range(len(pred))
                    ]
            prob = probs.data.cpu().tolist()
            #logger.debug('torch pred')
            #logger.debug(pred)
            #logger.debug(prob)
            final_pred.extend(pred_class)
            final_prob.extend(prob)
        return final_pred, final_prob


class Classifier_onnx(object):
    """Classifier for marque et modèle

    Classifies images using a pretrained model.
    """

    @timeit
    def __init__(self):
        self.session = onnxruntime.InferenceSession("classifcation_model.onnx")
        self.output_name = self.session.get_outputs()[0].name
        self.input_name = self.session.get_inputs()[0].name


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

        X = list()
        ind = 0
        for img, boxes in selected_boxes:
            logger.debug('img')
            logger.debug(img.size)
            logger.debug('MIDE')
            logger.debug(img.mode)
            detect_faces(np.array(img.crop(boxes)), ind)
            ind += 1
            #img = np.array(img.crop(boxes).resize((224, 224), Image.BILINEAR)).reshape(3, 224, 224).astype(np.float32)
            img = np.moveaxis(np.array(img.crop(boxes).resize((224, 224),
                Image.BILINEAR)), 2, 0).astype(np.float32)
            #img = np.array(Image.open(io.BytesIO(img.crop(boxes).resize((224, 224), Image.BILINEAR).tobytes())))
            #img = img.reshape(3, 224, 224).astype(np.float32)
            #logger.debug()
            img /= 255
            img -= np.array([0.485, 0.456, 0.406])[:, None, None]
            img /= np.array([0.229, 0.224, 0.225])[:, None, None]
            logger.debug(img.shape)
            logger.debug('box')
            logger.debug(boxes)
            X.append(img)

        #logger.debug(X)
        #logger.debug(np.array(X))
        logger.debug(len(X))
        logger.debug(np.array(X).shape)
        logger.debug(np.array(X).dtype)
        logger.debug(type(np.array(X)))
        logger.debug(np.array(X)[0])
        res = self.session.run([self.output_name], {self.input_name: np.array(X)})
        logger.debug('res.shape')
        logger.debug(len(res))
        norm_output = softmax(res[0])
        pred = np.argmax(norm_output, axis=1)
        prob = np.max(norm_output, axis=1)

        final_pred = list([[all_categories[str(i)]] for i in pred])
        final_prob = list([[i] for i in prob])

        ## Crop and resize
        #crop = Crop()
        #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                 std=[0.229, 0.224, 0.225])
        #preprocess = transforms.Compose([
        #    crop,
        #    transforms.Resize([224, 224]),
        #    transforms.ToTensor(),
        #    normalize,
        #])

        #val_loader = torch.utils.data.DataLoader(
        #        DatasetList(selected_boxes, transform=preprocess),
        #        batch_size=256, shuffle=False)

        #final_pred: List = list()
        #final_prob: List = list()
        #for inp in val_loader:
        #    if device.type == 'cuda':
        #        inp = inp.cuda(device, non_blocking=True)

        #    logger.debug(inp.shape)
        #    logger.debug(inp.data.numpy().shape)
        #    logger.debug(inp.data.numpy().dtype)
        #    logger.debug(type(inp.data.numpy()))
        #    logger.debug(inp.data.numpy()[0])
        #    res = self.session.run([self.output_name], {self.input_name: inp.data.numpy()})
        #    logger.debug('torch')
        #    logger.debug('onnx')
        #    logger.debug(res[0].shape)
        #    logger.debug('pred')
        #    norm_output = softmax(res[0])
        #    pred_onnx = np.argmax(norm_output, axis=1)
        #    prob_onnx = np.max(norm_output, axis=1)
        #    final_pred.extend([[all_categories[str(i)]] for i in pred_onnx])
        #    final_prob.extend([[i] for i in prob_onnx])

        return final_pred, final_prob
