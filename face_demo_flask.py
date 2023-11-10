import cv2
import time
from flask_demo.flask_camera import FlaskCamera
#from flask_camera import FlaskCamera

from utils.model_provider import ModelProvider
from utils.akida_worker import AkidaWorker
from utils.image_utils import FaceDetector

from flask_socketio import SocketIO, emit


def draw_detection(image, face_rect):
    """
    Draw detections bounding boxes in displayed image.

    args:
        image(cv.OutputArray): displayed image
        bounding_boxes(list): bounding boxes coordinates ordered by a dict
    """
    color = (128, 0, 255)

    # Draw bounding box and object name
    f_x, f_y, f_w, f_h = face_rect
    cv2.rectangle(image, (f_x, f_y), (f_x + f_w, f_y + f_h), color, 3)


# Overriding main class to grab and process frame
class FaceDemoFlask(FlaskCamera):

    def __init__(self, socket, hw_enabled):
        # Download model
        model_url = "http://data.brainchip.com/models/mobilenet_edge/mobilenet_imagenet_224_alpha_50_edge_iq8_wq4_aq4.h5"
        self.model_provider = ModelProvider()
        self.model_file = self.model_provider.fetch_model(model_url=model_url,
                                                          input_is_image=True)
        model_file_edge = self.model_provider.prepare_edge_learning(
            model_path=self.model_file,
            num_classes=5,
            num_neurons=1,
            num_weights=500)

        # Start worker and set model
        self.hw_enabled = hw_enabled
        self.worker = AkidaWorker()
        self.worker.start()
        self.worker.set_model(model_path=model_file_edge,
                              enable_hw=self.hw_enabled)
        self.worker.set_predict(num_classes=5)

        self.persons = []
        self.max_nb_shots = 5
        self.persons_nb_shots = [5, 5, 5, 5, 5]
        self.pred = ""
        self.power_label = ""
        self.fps = ""
        self.face_detector = FaceDetector()
        self.time_last_face_detection = time.time()

        self.socket = socket

        self.video_source = 0
        super(FaceDemoFlask, self).__init__()

    def set_video_source(self, source):
        self.video_source = source

    def frames(self):
        camera = cv2.VideoCapture(self.video_source)
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        while True:
            _, img = camera.read()

            face_rect = self.face_detector.detect(img)
            if face_rect is not None:
                self.time_last_face_detection = time.time()
                draw_detection(img, face_rect)

                x, y, w, h = face_rect
                im_crop = img[y:y + h, x:x + w]

                dim = (96, 112)
                resized = cv2.resize(im_crop,
                                     dim,
                                     interpolation=cv2.INTER_CUBIC)

                # Send command to worker and get wait for result
                self.worker.fetch_data(data=resized)
                try:
                    output = self.worker.get_output(block=False)

                    power_data = output[1]
                    if power_data is not None:
                        self.power_label = "Average power: %.0fmW" % (
                            power_data['avg_power'])
                        self.fps = "fps: %.2f" % (power_data['fps'])

                    if output[0].size == 1:
                        if self.persons and output[0] < len(self.persons):
                            self.pred = str(self.persons[output[0]])
                            self.pred += "<sup>" + str(
                                self.max_nb_shots -
                                self.persons_nb_shots[self.persons.index(
                                    self.pred)]) + "shots</sup>"
                        else:
                            self.pred = ""
                    else:
                        self.pred = ""
                except:
                    pass
            else:
                if time.time() - self.time_last_face_detection > 3:
                    self.pred = ""

            # Send message to enable/disable learn button
            # if a face is detected
            if face_rect is not None:
                self.socket.emit('learn_face', {'enable': True},
                                 namespace="/test")
            else:
                self.socket.emit('learn_face', {'enable': False},
                                 namespace="/test")

            yield cv2.imencode(
                '.jpg', img)[1].tobytes(), self.pred, self.power_label, self.fps

    def learn_class(self, person_name):
        if person_name not in self.persons:
            self.persons.append(person_name)
        shots = self.persons_nb_shots[self.persons.index(person_name)]
        if shots:
            self.persons_nb_shots[self.persons.index(person_name)] -= 1

        self.worker.set_learning(
            do_learning=True, learned_class_id=self.persons.index(person_name))

    def reset_model(self, max_persons, max_shots, neurons):
        print("[EDGE DEMO] Resetting model")
        self.persons = []
        self.persons_nb_shots = []
        self.max_nb_shots = max_shots
        for i in range(max_persons):
            self.persons_nb_shots.append(max_shots)
        model_file_edge = self.model_provider.prepare_edge_learning(
            model_path=self.model_file,
            num_classes=max_persons,
            num_neurons=neurons,
            num_weights=500)

        self.worker.set_model(model_path=model_file_edge,
                              enable_hw=self.hw_enabled)
        self.worker.set_predict(num_classes=max_persons)
