import cv2
import time
import numpy as np
from utils.flask_camera import FlaskCamera
from utils.model_provider import ModelProvider
from utils.akida_worker import AkidaWorker

# from flask_socketio import SocketIO, emit

#index_to_label =  {  0: "indu", 1:"nida",2:"sudhansu"}
#index_to_label_face =  { 0: 'indu', 1: 'narasimha', 2: 'nida', 3: 'raji', 4: 'sudhansu', 5:'nida'}
index_to_label_face = {0: 'AJN', 1: 'Chakri', 2: 'Divya', 3: 'Mahathi', 4: 'Narsimha', 5: 'Nida', 6: 'Rajarajeswari', 7: 'Sudhanshu', 8: 'Sushanth'}
#index_to_label =  {0: 'Narasimha', 1: 'nida', 2: 'raji', 3: 'sudhansu' }

index_to_label_emotion = {0: "Angry",  1: "Happy", 2: "Neutral", 3: "Sad"}
index_to_label_emotion = {0: 'angry', 1: 'happy', 2: 'neutral', 3: 'sad', 4: 'surprise'}
"""
    ######################################################################################################
    Face detection using opencv Haar cascade written in class 

"""


class FaceDetector():
    """The ``FaceDetector`` class construct a detector to looking for faces in
    the images. Algorithm used is OpenCV.
    """

    def __init__(self):
        self.detector = cv2.CascadeClassifier(
            'utils/haarcascade_frontalface_default.xml')

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
        gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Set minimal face size
        img_size = (image.shape[0] + image.shape[1]) // 2
        size_min = img_size // 8
        # Detect faces
        rects = self.detector.detectMultiScale(gray_img,
                                               scaleFactor=1.3,
                                               minNeighbors=5,
                                               #minSize=(size_min, size_min)
                                               )
        # Keep the bigger face if several faces are found
        for (x, y, w, h) in rects:
            if w + h > x2 - x1 + y2 - y1:
                x1 = x
                x2 = x + w
                y1 = y
                y2 = y + h

        if x2 - x1 > 0:
            return (x1, y1, x2 - x1, y2 - y1)

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
        file1 = open('./Combined/config.py','r')
        face = file1.read()
        print(face)
        self.face = int(face[0])

        # Download model
        
        # model_url = "http://data.brainchip.com/models/mobilenet/mobilenet_imagenet_224_alpha_50_iq8_wq4_aq4.h5"
        if self.face == 1:
            model_file = 'Models_akida/Akida_9.h5'
            #model_file = 'Models_akida/vgg_6hyd_tl3apr2_akida.h5'
        else:
            model_file = 'Models_akida/model_4cls_q_akida.h5'
            #model_file = 'Models_akida/FER_5_bkup_d12.h5'
            model_file = 'Models_akida/FER_5_bkup2.h5'
        # self.model_provider = ModelProvider()
        # model_file = self.model_provider.fetch_model(model_url=model_url,
                                                    #   input_is_image=True)

        # Start worker and set model
        self.worker = AkidaWorker()
        self.worker.start()
        self.worker.set_model(model_path=model_file, enable_hw=hw_enabled)
        self.worker.set_predict(int(3))

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
        fps = 0

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
                if self.face == 1:
                    resized = cv2.resize(img[y:y + h, x:x + w], (96, 112))
                    #resized = np.expand_dims(cv2.resize(cv2.cvtColor(img[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY), (96, 112)), -1)
                else:
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

                    
                    if power_data is not None:
                        self.power_label = "Average power: %.0fmW" % (
                            power_data['avg_power'])
                        #self.fps = "fps: %.2f" % (power_data['fps'])
                    
                    result = np.argmax(output[0])
                    
                    print('OUTPUT:',output[0])
                    if self.face == 1:
                        index_to_label = index_to_label_face
                    else:
                        index_to_label = index_to_label_emotion
                    self.prediction = index_to_label[result]
                    # print("###################################")
                    print("prediction",self.prediction)
                    end = time.time()
                    fps = 1 / (end - new_time)
                    
                    #Fps = "%.0f" % (self.fps)
                    #x,y,_ = img.shape
                    
                except Exception as e:
                    print(e)
                    pass
                cv2.putText(img, self.prediction, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(img, self.power_label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(img, str("FPS: %.0f" % (fps)), (wth, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                yield cv2.imencode('.jpg', img)[1].tobytes(), self.prediction, self.power_label, self.fps

