import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import onnxruntime

model_name = 'faster_rcnn_r50_c4_2x-6e4fdf4f'#'ssd512_coco_vgg16_caffe_120e_20181221-d48b0be8'
pytorch_model = '/model/%s.pth' % model_name
onnx_model = '/model/%s.onnx'

# export to onnx
def forward_dummy(self, img):
    x = self.extract_feat(img)
    x = self.bbox_head(x)
    return x

# try to infer onnx model
X = list()
img = cv2.imread("./clio4.jpg")
img.resize((3,512,512))
print("img shape : " + str(img.shape))
img= np.array(img).astype(np.float32)
img /= 255
img -= np.array([0.485, 0.456, 0.406])[:, None, None]
img /= np.array([0.229, 0.224, 0.225])[:, None, None]
X.append(img)

session = onnxruntime.InferenceSession(onnx_model)
output_name = session.get_outputs()[0].name
input_name = session.get_inputs()[0].name
res = session.run([output_name], {input_name: np.array(X)})
