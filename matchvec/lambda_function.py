from os import listdir, path, getenv
import json
import numpy as np
import cv2
import onnxruntime
from PIL import Image
import base64
from cgi import parse_header, parse_multipart
from io import BytesIO
from process import predict_class, predict_objects

def lambda_handler(event, context):
    print("ENV", getenv('BACKEND'))
    print("ENV", getenv('DETECTION_THRESHOLD'))
    print("LISTDIR", listdir('/tmp'))

    res = list()
    return {
        'statusCode': 200,
        'body': json.dumps(res)
    }


def lambda_handler_classification(event, context):
    print("ENV", getenv('BACKEND'))
    print("ENV", getenv('DETECTION_THRESHOLD'))
    print("LISTDIR", listdir('/tmp'))

    res = list()
    body_str = event.get('body', None)
    body_parsed = json.loads(body_str).get('body', None)

    #c_type, c_data = parse_header(event['headers']['Content-Type'])
    #assert c_type == 'multipart/form-data'
    #decoded_string = base64.b64decode(event['body'])
    #form_data = parse_multipart(BytesIO(decoded_string), c_data)

    if body_parsed:
        print(type(body_parsed))
        print(body_parsed[:100])

        #  read encoded image
        imageString = base64.b64decode(body_parsed)
        #  convert binary data to numpy array
        nparr = np.frombuffer(imageString, np.uint8)
        #  let opencv decode image to correct format
        img = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)

        res.append(predict_class(img))

    return {
        'statusCode': 200,
        'body': json.dumps(res)
    }
