import sys, os
import cv2
import pytest
import matchvec
import io

print(10*"/")
from requests_toolbelt.multipart.encoder import MultipartEncoder
# Import lambda_function outside pâckage matchvec
print("PATH")
print(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__),'aws_lambda'))

from aws_lambda.lambda_function import lambda_handler_classification

def test_handler(apigateway_event):
    resp = lambda_handler_classification(apigateway_event, None)
    body = resp['body']
    assert resp['statusCode'] == 200
    print(body)
    
    assert any(['CLIO' in vehicule['brand_model_classif']['label'] for vehicule in eval(body)[0] + eval(body)[1]]), 'There is no clio in predictions %s'%body
    assert 'BMW SERIE 5' in body, 'There is no bmw in predictions %s'%body
    if 'CLASSIFICATION_MODEL_PRIO' in os.environ:
        assert any(['AUTRES' in vehicule['prio_classif']['label'] for vehicule in eval(body)[0] + eval(body)[1]]), 'There is no autres in predictions %s'%body
    else:
        print('!!!! Test not executed, add CLASSIFICATION_MODEL_PRIO path !!!!!')


def test_preflight():
    event = dict(httpMethod = 'OPTIONS',
                 path = '/object_detection',
                 )
    resp = lambda_handler_classification(event, None)
    print(resp)
    assert resp['statusCode'] == 200

if __name__ == '__main__':
    mp_encoder = MultipartEncoder(
        fields={'field0': open("tests/binary1.dat", "rb"),
                'field1': open("tests/binary2.dat", "rb")}
    )
    mp_encoder = MultipartEncoder(
        fields={'field0': open("tests/clio4.jpg", "rb")}
    )
    body = mp_encoder.to_string()
    event = dict(httpMethod = 'POST',
                 path = '/predict',
                 headers = {'Content-Type': mp_encoder.content_type},
                 body = body)

    test_handler(event)
