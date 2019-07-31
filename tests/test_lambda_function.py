import sys, os
import cv2
import unittest
import base64

sys.path.append('./matchvec')
from lambda_function import lambda_handler

class TestFileFail(unittest.TestCase):

    def test_handler(self):

        with open("tests/clio-peugeot.jpg", "rb") as imageFile:
            img = base64.b64encode(imageFile.read())
        event = {
                'body': img
                }
        lambda_handler(event, None)


    #def test_request_prediction(self):
    #    data = open('tests/clio-peugeot.jpg', 'rb').read()

    #    with app.test_client() as c:
    #        resp = c.post(
    #            '/predict',
    #            content_type='image/jpg',
    #            body=data
    #        )
    #        self.assertEqual(
    #            '200 OK',
    #            resp.status,
    #            )

if __name__ == '__main__':
    unittest.main()
