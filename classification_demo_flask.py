import cv2
from flask_demo.flask_camera import FlaskCamera
import numpy as np

from utils.model_provider import ModelProvider
from utils.akida_worker import AkidaWorker

from akida_models.imagenet.preprocessing import index_to_label


# Overriding main class to grab and process frame
class ClassificationDemoFlask(FlaskCamera):

    def __init__(self, hw_enabled):
        # Download model
        model_url = "http://data.brainchip.com/models/mobilenet/mobilenet_imagenet_224_alpha_50_iq8_wq4_aq4.h5"
        self.model_provider = ModelProvider()
        model_file = self.model_provider.fetch_model(model_url=model_url,
                                                     input_is_image=True)

        # Start worker and set model
        self.worker = AkidaWorker()
        self.worker.start()
        self.worker.set_model(model_path=model_file, enable_hw=hw_enabled)

        self.prediction = ""
        self.power_label = ""
        self.fps = ""
        self.video_source = 0
        super(ClassificationDemoFlask, self).__init__()

    def set_video_source(self, source):
        self.video_source = source

    def frames(self):
        camera = cv2.VideoCapture(self.video_source)
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        while True:
            _, img = camera.read()

            dim = (224, 224)
            resized = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)

            # Send command to worker and get wait for result
            self.worker.fetch_data(data=resized)
            try:
                output = self.worker.get_output(block=False)

                power_data = output[1]
                if power_data is not None:
                    self.power_label = "Average power: %.0fmW" % (
                        power_data['avg_power'])
                    self.fps = "fps: %.2f" % (power_data['fps'])

                result = np.argmax(output[0])

                self.prediction = index_to_label(result)
            except:
                pass

            yield cv2.imencode(
                '.jpg',
                img)[1].tobytes(), self.prediction, self.power_label, self.fps
