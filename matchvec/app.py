import os
import json
import cv2
import numpy as np
from flask import Flask, send_from_directory, request, Blueprint, url_for
from flask_restplus import Resource, Api, reqparse, fields
from matchvec.process import predict_class, predict_objects
from werkzeug.datastructures import FileStorage
from urllib.request import urlopen
from matchvec.utils import logger
from celery import Celery


app = Flask(__name__)
app.config.SWAGGER_UI_DOC_EXPANSION = 'list'
app.config.SWAGGER_UI_OPERATION_ID = True
app.config.SWAGGER_UI_REQUEST_DURATION = True

# Celery configuration
app.config['CELERY_BROKER_URL'] = 'redis://redis:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://redis:6379/0'

# Initialize Celery
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)


######################
#  long task celery  #
######################


@celery.task(bind=True)
def long_task(self, video_name):
    #video = request.files.getlist('video', None)
    logger.debug(video_name)
    res = list()
    #if video:

    #logger.debug("Filename {}".format(video[0].filename))
    #video[0].save("/tmp/video")
    cap = cv2.VideoCapture("/tmp/video", )
    while not cap.isOpened():
        cap = cv2.VideoCapture("/tmp/video", )
        cv2.waitKey(1000)
        print("Wait for the header")

    pos_frame = cap.get(cv2.cv2.CAP_PROP_POS_FRAMES)
    while True:
        flag, frame = cap.read()
        if flag:
            # The frame is ready and already captured
            pos_frame = int(cap.get(cv2.cv2.CAP_PROP_POS_FRAMES))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.update_state(state='PROGRESS',
                              meta={
                                  'current': pos_frame,
                                  'total': total_frames,
                                  'partial_result': res
                                  })
            h, w,  _ = frame.shape
            frame = frame[0:h,int(2*w/3):w]
            #frame = frame[0:h,0:w]
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            #retval, buff = cv2.imencode('.jpg', frame)
            #logger.debug(h)
            #logger.debug(w)
            output = predict_class(frame)
            if pos_frame == 0:
                cv2.imwrite('imgtest1sur{}.jpg'.format(total_frames), frame)
            if (pos_frame%50 == 0):
                cv2.imwrite('imgtest{}.jpg'.format(pos_frame), frame)
            #res = requests.post(url, files=files)
            if len(output) > 0:
                for box in output:
                    logger.debug('Frame {}'.format(pos_frame))
                    logger.debug(box)
                    if float(box['confidence']) > 0.50 and float(box['prob'][0]) > 0.85:
                        logger.debug(box['pred'][0])
                        res.append({pos_frame: box['pred'][0]})
                        cv2.imwrite('imgtest{}.jpg'.format(pos_frame), frame)
        else:
            break
    #return res
    return {'current': total_frames, 'total': total_frames, 'status': 'Task completed!',
            'result': res}


@app.route('/matchvec/status/<task_id>')
def taskstatus(task_id):
    task = long_task.AsyncResult(task_id)
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'current': 0,
            'total': 1,
            'status': 'Pending...'
        }
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'current': task.info.get('current', 0),
            'total': task.info.get('total', 1),
            'status': task.info.get('status', '')
        }
        if 'result' in task.info:
            response['result'] = task.info['result']
    else:
        # something went wrong in the background job
        response = {
            'state': task.state,
            'current': 1,
            'total': 1,
            'status': str(task.info),  # this is the exception raised
        }
    return json.dumps(response)


##########################
#  Documentation Sphinx  #
##########################

blueprint_doc = Blueprint('documentation', __name__,
                          static_folder='../docs/build/html/_static',
                          url_prefix='/matchvec/docs')


@blueprint_doc.route('/', defaults={'filename': 'index.html'})
@blueprint_doc.route('/<path:filename>')
def show_pages(filename):
    return send_from_directory('../docs/build/html', filename)


app.register_blueprint(blueprint_doc)

#################
#  API SWAGGER  #
#################


class Custom_API(Api):
    @property
    def specs_url(self):
        '''
        The Swagger specifications absolute url (ie. `swagger.json`)

        :rtype: str
        '''
        return url_for(self.endpoint('specs'), _external=False)


blueprint = Blueprint('api', __name__, url_prefix='/matchvec')
api = Custom_API(blueprint, doc='/swagger', version='1.0', title='IA Flash',
                 description='Classification marque et modèle')
app.register_blueprint(blueprint)


parser = reqparse.RequestParser()
parser.add_argument('url', type=str, location='form', help='Image URL in jpg format. URL must end with jpg.')
parser.add_argument('image', type=FileStorage, location='files', help='Image saved locally. Multiple images are allowed.')

parser_video = reqparse.RequestParser()
parser_video.add_argument('video', type=FileStorage, location='files', help='Video used for image analysis.')


BaseOutput = api.model('BaseOutput', {
    'x1': fields.Integer(description='X1', min=0, example=10),
    'y1': fields.Integer(description='Y1', min=0, example=10),
    'x2': fields.Integer(description='X2', min=0, example=200),
    'y2': fields.Integer(description='Y2', min=0, example=200),
    'class_name': fields.String(description='Object detection label',
                                example='car'),
    'confidence': fields.Float(description='Object detection confidence score',
                               min=0, max=1, example=0.95),
    })


ObjectDetectionOutput = api.inherit('ObjectDetectionOutput', BaseOutput, {
            'label': fields.String(
                description='Object detection label for visualization',
                example='car: 0.95'),
            })


ClassificationOutput = api.inherit('ClassificationOutput', BaseOutput, {
            'label': fields.String(
                description='Classification label for visualization',
                example='PEUGEOT 207: 0.54'),
            'pred': fields.List(fields.String(),
                description='5 first predictions classes',
                example=['PEUGEOT 207', 'CITROEN C3 PICASSO', 'NISSAN MICRA',
                         'CITROEN XSARA PICASSO', 'RENAULT KANGOO']
                ),
            'prob': fields.List(fields.Float(),
                description='5 first prediction probabilities',
                example=[0.5462563633918762, 0.07783588021993637, 0.047950416803359985,
                    0.041797831654548645, 0.03768396005034447])
                })


@api.route('/video_detection')
class VideoDetection(Resource):
    """Docstring for MyClass. """

    @api.expect(parser_video)
    def post(self):
        """Video detection"""
        video = request.files.getlist('video', None)
        logger.debug(video)
        res = list()
        if video:
            video[0].save("/tmp/video")
            task = long_task.delay(video[0].filename)
            return {'task_id': task.id}, 202
        else:
            return {'status': 'no video'}, 404


@api.route('/object_detection')
class ObjectDetection(Resource):
    """Docstring for MyClass. """

    @api.expect(parser)
    @api.marshal_with(ObjectDetectionOutput, mask=None, as_list=True, description='Result is a list of the following element (double list)')
    def post(self):
        """Object detection

        Image can be loaded either by using an internet URL in the url field or
        by using a local stored image in the image field
        """
        images = request.files.getlist('image', None)
        url = request.form.get('url', None)
        res = list()
        if url:
            try:
                resp = urlopen(url)
                img = np.asarray(bytearray(resp.read()), dtype="uint8")
                img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
                res.append(predict_objects(img))
            except Exception as e:
                print(url)
                print(e)
        if images:
            for i in range(len(images)):
                nparr = np.frombuffer(images[i].read(), np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
                res.append(predict_objects(img))
        return res


@api.route('/predict')
class ClassPrediction(Resource):
    """Predict vehicule class"""

    @api.expect(parser)
    @api.marshal_with(ClassificationOutput, mask=None, as_list=True, description='Result is a list of the following element (double list)')
    def post(self):
        """Brand and model classifcation

        Image can be loaded either by using an internet URL in the url field or
        by using a local stored image in the image field
        """
        images = request.files.getlist('image')
        url = request.form.get('url', None)

        res = list()
        if url:
            try:
                resp = urlopen(url)
                img = np.asarray(bytearray(resp.read()), dtype="uint8")
                img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
                res.append(predict_class(img))
            except Exception as e:
                print(url)
                print(e)
        if images:
            for i in range(len(images)):
                nparr = np.frombuffer(images[i].read(), np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
                res.append(predict_class(img))
        return res


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=bool(os.getenv('DEBUG')))
