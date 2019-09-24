import sys, os
import cv2
import pytest
import base64
import matchvec
import io

def test_predict(app, file_clio4):

    with app.test_client() as c:
        resp = c.post(
            '/matchvec/predict',
            content_type = 'multipart/form-data',
            data = file_clio4(10))

        print(resp.get_data(as_text=False)[0])

    resp_data = eval(resp.get_data(as_text=True))

    assert resp.status_code == 200, 'Status Code : %s'%resp.status_code

    assert resp.status_code == 200, 'Status Code : %s'%resp.status_code
    assert resp_data[0][0]["prob"][0] > 0.8, 'Classif confidence too low'
    assert resp_data[0][0]["pred"][0] == "RENAULT CLIO",  'Not a clio'
    assert resp_data[0][0]["confidence"] > 0.8, "Car not detected"

if __name__ == '__main__':
    test_predict()
