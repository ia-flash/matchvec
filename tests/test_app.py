import sys, os
import cv2
import pytest
import base64
import matchvec
import io


def assert_clio(resp_data) :
    assert resp_data[0][0]["brand_model_classif"]["prob"][0] > 0.8, 'Classif confidence too low %s'%resp_data
    assert resp_data[0][0]["brand_model_classif"]["pred"][0] == "RENAULT CLIO",  'Not a clio'
    assert resp_data[0][0]["confidence"] > 0.8, "Car not detected"

def assert_pompier(resp_data) :
    assert resp_data[0][0]["prio_classif"]["prob"][0] > 0.65, 'Classif confidence too low %s'%resp_data
    assert resp_data[0][0]["prio_classif"]["pred"][0] == "POMPIER",  'Not a fire fighter'
    assert resp_data[0][0]["confidence"] > 0.7, "Car not detected"

def test_predict_image(app, file_clio4):

    with app.test_client() as c:
        resp = c.post(
            '/matchvec/predict',
            content_type = 'multipart/form-data',
            data = file_clio4(5))
        print(resp.get_data(as_text=False)[0])

    # Test with an image
    assert resp.status_code == 200, 'Status Code : %s'%resp.status_code

    resp_data = eval(resp.get_data(as_text=True))
    assert_clio(resp_data)

def test_predict_url(app, url_clio):
    with app.test_client() as c:
        resp = c.post(
            '/matchvec/predict',
            content_type = 'multipart/form-data',
            data = {'url':url_clio})
        print(resp.get_data(as_text=False)[0])

    # Test with an image
    resp_data = eval(resp.get_data(as_text=True))
    assert_clio(resp_data)


def test_predict_prio_url(app, url_pompier):
    with app.test_client() as c:
        resp = c.post(
            '/matchvec/predict',
            content_type = 'multipart/form-data',
            data = {'url':url_pompier})
        print(resp.get_data(as_text=False)[0])

    # Test with an image
    resp_data = eval(resp.get_data(as_text=True))
    assert_pompier(resp_data)

def test_predict_anonym_url(app, url_anonym):
    with app.test_client() as c:
        resp = c.post(
            '/matchvec/anonym',
            content_type = 'multipart/form-data',
            data = {'url':url_anonym})
        print(resp.get_data(as_text=False)[0])

    # Test with an image
    resp_data = eval(resp.get_data(as_text=True))
    assert sum(['plate' in obj['label'] for obj in resp_data[0]]) == 1, 'Some plates are missing'
    assert sum(['person' in obj['label'] for obj in resp_data[0]]) == 1, 'Some persons are missing'
