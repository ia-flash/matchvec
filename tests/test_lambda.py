import sys, os
import cv2
import pytest
import base64
import matchvec
import io
from matchvec.lambda_function import lambda_handler_classification


def test_handler(event_clio4):

    resp = lambda_handler_classification(event_clio4, None)
    assert resp['statusCode'] == 200

if __name__ == '__main__':
    test_handler()
