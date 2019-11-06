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
    assert any(['CLIO' in vehicule['label'] for vehicule in eval(body)[0] + eval(body)[1]]), 'There is no clio in first predictions'
    assert any(['BMW SERIE 5' in vehicule['label'] for vehicule in eval(body)[0] + eval(body)[1]]), 'There is no bmw in first predictions'

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
