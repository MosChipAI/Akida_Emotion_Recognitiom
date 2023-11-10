import os, time
import sys
import numpy as np

from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit

# from flask_demo.classification_demo_flask import ClassificationDemoFlask
# from flask_demo.detection_demo_flask import DetectionDemoFlask
# from flask_demo.edge_demo_flask import EdgeDemoFlask 
#from Combined.face_demo_flask import FaceDemoFlask
from Combined.New_script_v1 import ClassificationDemoFlask
from Combined import config
import akida

# Default port for Flask app
HTTP_PORT = 5000

# Init Flask app and SocketIO and declare global camera variable
app = Flask(__name__,
            static_folder='Combined',
            static_url_path='/Combined',
            template_folder='Combined/templates')
socketio = SocketIO(app)
camera = None
hw_enabled = False


@app.route('/')
def index():
    """
    Demo application home page.
    """
    global hw_enabled
    return render_template('index.html',
                           demo_app_title=sys.argv[1],
                           hw_enabled=hw_enabled)


def gen(camera):
    """
    Video streaming generator function.
    """
    while True:
        frame, res, pw_stats, fps = camera.get_frame()
        socketio.emit('newlabel', {'number': res}, namespace="/test")
        socketio.emit('hw_data', {
            'pw_stats': pw_stats,
            'fps': fps
        },
                      namespace="/test")
        yield (b'--frame\r\n'
               b'Content-Type: image/png\r\n\r\n' + frame + b'\r\n')


@app.route('/videoel')
def video_feed():
    """
    Video streaming route.
    Put this in the src attribute of an img tag.
    """
    global camera
    print("AkidaDemo: Start Video feed")
    return Response(gen(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@socketio.on('connect', namespace='/test')
def test_connect():
    print('AkidaDemo: Web client connected')


@socketio.on('disconnect', namespace='/test')
def test_disconnect():
    print('AkidaDemo: Web client disconnected')


@socketio.on('request', namespace='/test')
def test_update_settings(data):
    global camera
    print("AkidaDemo: Message receive :" + str(data))
    msg_type = data.get('type')
    msg_data = str(data.get('data')).strip()

    if (msg_type == "settings"):
        camera.set_model(msg_data)
        socketio.emit('load_finished', None, namespace="/test")
    elif msg_type == "learning":
        if str(msg_data).strip() == "reset":
            camera.reset_model(int(data.get("max_classes")),
                               int(data.get("max_shots")),
                               int(data.get("neurons")))
        else:
            camera.learn_class(msg_data)


if __name__ == '__main__':
    if not len(sys.argv) >= 2:
        raise RuntimeError("Missing first positional argument - Demo Name")
        sys.exit()

    # Check if an Akida device is available. If yes, first akida device will be
    # selected by default.
    hw_enabled = False
    if akida.devices():
        if akida.devices()[0].version == akida.NSoC_v1:
            print(
                "AkidaDemo: NSoC_v1 is no longer supported. Demo will run in SW."
            )
        else:
            hw_enabled = True
            if sys.argv[1] == "Classification Demo":
                camera = ClassificationDemoFlask(hw_enabled=hw_enabled)
            elif sys.argv[1] == "Detection Demo":
                camera = DetectionDemoFlask(hw_enabled=hw_enabled)
            elif sys.argv[1] == "Edge Demo":
                camera = EdgeDemoFlask(hw_enabled=hw_enabled)
    file1 = open('./Combined/config.py','w')
    if sys.argv[1] == "Face Demo" or sys.argv[1] == "Emotion Demo":
        if sys.argv[1] == "Face Demo":
            file1.write("1")
        else:
            file1.write("0")
        file1.close()
        camera = ClassificationDemoFlask(hw_enabled=hw_enabled)
        #camera = FaceDemoFlask(socket=socketio,hw_enabled=hw_enabled)
    else:
        raise RuntimeError("Unknown Akida Demo name")
        sys.exit()

    # Init Camera object (USB)
    os.environ["CAP_PROP_FRAME_WIDTH"] = "620"
    os.environ["CAP_PROP_FRAME_HEIGHT"] = "480"
    camera.set_video_source([0])

    socketio.on_event('request', test_update_settings, namespace='/test')
   # socketio.run(app, debug=False, host='192.168.200.97', port=5000)
    socketio.run(app, debug=False, host='10.42.0.1', port=5000)
    #socketio.run(app, debug=False, host='0.0.0.0', port=5000)
