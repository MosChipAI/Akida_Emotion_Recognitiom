import cv2
import time
import numpy as np
from utils.flask_camera import FlaskCamera
from utils.model_provider import ModelProvider
from utils.akida_worker import AkidaWorker
import mediapipe as mp

# from flask_socketio import SocketIO, emit

index_to_label = {0: "CC", 1: "BB", 2: "DD", 3: "EE", 4: "FF", 5: "GG"}
#index_to_label =  {  0: "indu", 1:"nida",2:"sudhansu"}
"""
    ######################################################################################################
    Face detection using opencv Haar cascade written in class 

"""


class FaceDetector():
    """The ``FaceDetector`` class construct a detector to looking for faces in
    the images. Algorithm used is OpenCV.
    """

    def __init__(self):
        # Initialize the face detection model
        self.face_detection = mp.solutions.face_detection.FaceDetection()

    def detect(self, image):
        """Returns the position of the bigger face in the image.

        Args:
            image (:obj:`numpy.ndarray`): input image.

        Returns:
            tuple: a tuple containing the top left coordinates and dimensions of
                the face found (x, y, w, h). Or None if no face was found in the
                image.
        """
        # Coordinate of the face found
        x1 = x2 = y1 = y2 = 0

        # Convert color image to grey scale
        #gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Set minimal face size
        img_size = (image.shape[0] + image.shape[1]) // 2
        size_min = img_size // 8
        # Detect faces
        # Detect faces
        results = self.face_detection.process(image)
        # Keep the bigger face if several faces are found
        if results.detections:
            #print(f"Total detected faces : {len(results.detections)}")
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = image.shape
                xmin = int(bbox.xmin * w)
                ymin = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                return (xmin,ymin,width,height)


        return None


########################################################################################
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
class ClassificationDemoFlask(FlaskCamera):

    def __init__(self, hw_enabled):
        # Download model
        # model_url = "http://data.brainchip.com/models/mobilenet/mobilenet_imagenet_224_alpha_50_iq8_wq4_aq4.h5"
        model_file = 'models/gray_faces_9.fbz'
        # self.model_provider = ModelProvider()
        # model_file = self.model_provider.fetch_model(model_url=model_url,
        #                                              input_is_image=True)

        # Start worker and set model
        self.worker = AkidaWorker()
        self.worker.start()
        self.worker.set_model(model_path=model_file, enable_hw=hw_enabled)
        self.worker.set_predict(int(6))

        self.prediction = ""
        self.power_label = ""
        self.face_detector = FaceDetector()
        self.fps = ""
        self.video_source = 0  
        super(ClassificationDemoFlask, self).__init__()

    def set_video_source(self, source):
        self.video_source = source
        #self.video_source = 0

    def frames(self):
        camera = cv2.VideoCapture(self.video_source)
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')
        fps=0
        while True:
            _, img = camera.read()
            wth, ht = img.shape[:-1]
            new_time = time.time()
            
            
            face_rect = self.face_detector.detect(img)
            if face_rect is not None:
                self.time_last_face_detection = time.time()
                draw_detection(img, face_rect)

                x, y, w, h = face_rect
                # im_crop = img[y:y + h, x:x + w]
                # gray = cv2.cvtColor(img[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
                resized = np.expand_dims(cv2.resize(cv2.cvtColor(img[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY), (98, 98)), -1)
                #print("RESIZED: ", resized.shape)
                #dim = (98, 98)
                #resized = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)

                # Send command to worker and get wait for result
                self.worker.fetch_data(data=resized)
                try:
                    output = self.worker.get_output(block=False)
                    #print("OUTPUT : ",output)
                    power_data = output[1]
                    #print("FPS : ",power_data['fps'])
                    if power_data is not None:
                        self.power_label = "Average power: %.0fmW" % (
                            power_data['avg_power'])
                        self.fps = "fps: %.2f" % (power_data['fps'])

                    result = np.argmax(output[0])

                    self.prediction = index_to_label[result]
                    # print("###################################")
                    print("prediction",self.prediction)
                    end = time.time()
                    fps = 1 / (end - new_time)
                    print('FPS:',fps)
                    #x,y,_ = img.shape
                except Exception as e:
                    print(e)
                    pass
                cv2.putText(img, self.prediction, (x, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(img, str("FPS: %.0f" % (fps)), (wth, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                yield cv2.imencode('.jpg', img)[1].tobytes(), self.prediction, self.power_label, self.fps

