import os
import json
import cv2
import base64
import numpy as np
import werkzeug
import logging
werkzeug.cached_property = werkzeug.utils.cached_property
from flask import Flask, send_from_directory, request, Blueprint, url_for
from flask_restplus import Resource, Api, reqparse, fields
from matchvec.process import predict_class, predict_objects, predict_anonym
from werkzeug.datastructures import FileStorage
from urllib.request import urlopen

from celery import Celery
#from pymediainfo import MediaInfo

logger = logging.getLogger(__name__)

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


def rotate_frame90(image, number):
    for i in range(number):
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return image


@celery.task(bind=True)
def long_task(self, video_name, rotation90, prob_detection, prob_classification, selected_fps):
    logger.debug(video_name)
    res = dict()
    cap = cv2.VideoCapture("/tmp/video", )
    media_info = MediaInfo.parse('/tmp/video')
    myjson = json.loads(media_info.to_json())
    rotation = myjson['tracks'][1]['rotation']
    total_rotation = int(float(rotation)/90) + int(rotation90/90)
    logger.debug('Rotation total {}'.format(rotation))
    while not cap.isOpened():
        cap = cv2.VideoCapture("/tmp/video", )
        cv2.waitKey(1000)
        logger.debug("Wait for the header")

    pos_frame = cap.get(cv2.cv2.CAP_PROP_POS_FRAMES)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    skip_images = int(fps/selected_fps)
    logger.debug("Real fps {}, selected fps: {}, taking 1 image between {}".format(fps, selected_fps, skip_images))
    while True:
        flag, frame = cap.read()
        if flag:
            # The frame is ready and already captured
            pos_frame = int(cap.get(cv2.cv2.CAP_PROP_POS_FRAMES))
            # Every 10 frames
            if pos_frame % skip_images == 0:
                h, w,  _ = frame.shape
                # frame = frame[0:h,int(2*w/3):w]
                frame = frame[0:h, 0:w]
                frame = rotate_frame90(frame, total_rotation)
                self.update_state(state='PROGRESS',
                                  meta={
                                      'current': pos_frame,
                                      'total': total_frames,
                                      'partial_result': [{
                                          'frame': res[key]['frame'], 'seconds': res[key]['seconds'],
                                          'model': key, 'img': res[key]['img']} for key in res]
                                      })

                output = predict_class(frame)
                if len(output) > 0:
                    for box in output:
                        logger.debug('Frame {}'.format(pos_frame))
                        logger.debug(box)
                        if float(box['confidence']) > (prob_detection/100) and float(box['prob'][0]) > (prob_classification/100):
                            logger.debug(box['pred'][0])
                            # Print detected boxes
                            cv2.rectangle(frame, (box['x1'], box['y1']), (box['x2'], box['y2']), (255, 0, 0), 6)
                            cv2.putText(frame, box['label'], (box['x1'], box['y1'] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            # Convert captured image to JPG
                            retval, buffer = cv2.imencode('.jpg', frame)
                            # Convert to base64 encoding and show start of data
                            jpg_as_text = base64.b64encode(buffer)
                            base64_string = jpg_as_text.decode('utf-8')
                            modele = box['pred'][0]
                            res[modele] = {'frame': pos_frame, 'seconds': pos_frame/fps, 'model': box['pred'][0], 'img': base64_string}
        else:
            break
    return {'current': total_frames, 'total': total_frames, 'status': 'Task completed!',
            'partial_result': [{
                'frame': res[key]['frame'], 'seconds': res[key]['seconds'],
                'model': key, 'img': res[key]['img']} for key in res],
            'result': list(res.keys())}


@app.route('/matchvec/killtask/<task_id>')
def killtask(task_id):
    response = celery.control.revoke(task_id, terminate=True)
    return json.dumps(response)


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
            'status': task.info.get('status', ''),
            'partial_result': task.info.get('partial_result', list())
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
parser_video.add_argument('video', type=FileStorage, location='files',
                          help='Video used for image analysis.')


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

ClassifOutput = api.model('ClassifOutput', {
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

ObjectDetectionOutput = api.inherit('ObjectDetectionOutput', BaseOutput, {
            'label': fields.String(
                description='Object detection label for visualization',
                example='car: 0.95'),
            })

if 'CLASSIFICATION_MODEL_PRIO' in os.environ:
    ClassificationOutput = api.inherit('ClassificationOutput', BaseOutput, {
                'brand_model_classif': fields.Nested(ClassifOutput),
                'prio_classif': fields.Nested(ClassifOutput)})
else:
    ClassificationOutput = api.inherit('ClassificationOutput', BaseOutput, {
                'brand_model_classif': fields.Nested(ClassifOutput)})



@api.route('/video_detection', doc=False)
class VideoDetection(Resource):
    """Docstring for MyClass. """

    @api.expect(parser_video)
    def post(self):
        """Video detection"""
        video = request.files.getlist('video', None)
        rotation = int(request.form.get('rotation', 0))
        prob_detection = int(request.form.get('probDetection', 0))
        prob_classification = int(request.form.get('probClassification', 0))
        fps = int(request.form.get('fps', 0))
        logger.debug(video)
        logger.debug(rotation)
        if video:
            video[0].save("/tmp/video")
            task = long_task.delay(video[0].filename, rotation, prob_detection, prob_classification, fps)
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
                res.append([])

        if images:
            for i in range(len(images)):
                try :
                    nparr = np.frombuffer(images[i].read(), np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
                    res.append(predict_objects(img))
                except Exception as e:
                    print(e)
                    res.append([])
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
        logger.info("Détection et classification d'objets")
        images = request.files.getlist('image')
        url = request.form.get('url', None)
        res = list()
        if url:
            logger.info("Traitement image %s", url)
            try:
                resp = urlopen(url)
                img = np.asarray(bytearray(resp.read()), dtype="uint8")
                img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
                res.append(predict_class(img))
            except Exception as e:
                logger.exception(e)
                print(url)
                print(e)
                res.append([])

        if images:
            logger.info("Traitement de %d images", len(images))
            for i in range(len(images)):
                try:
                    nparr = np.frombuffer(images[i].read(), np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
                    res.append(predict_class(img))
                except Exception as e:
                    logger.exception(e)
                    print(e)
                    res.append([])

        logger.info("Résultats : %d objets trouvés", sum(map(len, res)))
        return res


@api.route('/anonym', doc=False)
class AnonymPrediction(Resource):
    """Image anonymisation"""

    @api.expect(parser)
    def post(self):
        """Anonymisation

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
                df = predict_anonym(img)
                if df is not None:
                    res.append(df)
            except Exception as e:
                print(url)
                print(e)
        if images:
            for i in range(len(images)):
                nparr = np.frombuffer(images[i].read(), np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
                df = predict_anonym(img)
                if df is not None:
                    res.append(df)
        return res



if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=bool(os.getenv('DEBUG')))
