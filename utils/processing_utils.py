import enum
import numpy as np

from utils.akida_power import AkidaPower

import akida
from akida import Model, LayerType
from akida.compatibility import create_from_model


class Command(enum.Enum):
    Reset = 0
    Evaluate = 1
    Fit = 2
    Predict = 3
    Stop = 4


def forward_frame(model, frame):
    return model.forward(np.expand_dims(frame, axis=0))


def evaluate_frame(model, frame):
    return model.evaluate(np.expand_dims(frame, axis=0)).squeeze()


def fit(model, frame, label):
    return model.fit(np.expand_dims(frame, axis=0), label)


def predict(model, frame, num_classes):
    prediction = model.predict(np.expand_dims(frame, axis=0), num_classes)

    return prediction[0]


def reset_model(model_path, hw_enabled):
    model = Model(model_path)
    if hw_enabled:
        if akida.devices()[0].version == akida.NSoC_v1:
            raise Exception("ProcessingUtils: NSoC_v1 is no longer supported.")
        else:
            model_comp = Model(layers=model.layers)

        try:
            model_comp.map(akida.devices()[0], hw_only=False)
            print("ProcessingUtils: Map model on " +
                  akida.devices()[0].desc.partition("/")[2])
        except Exception as e:
            
            raise Exception("ProcessingUtils: Unable to map model: " + str(e))

        return model_comp

    # Important to return the model
    return model


def worker_task(messages, outputs):
    """Processing task receiving commands from the application.

    The worker_task is spawned by the application at initialization to offload
    the Akida processing. It blocks on its input queue expecting messages made
    of a `Command` and its specific parameters. It possibly returns Akida
    processing outcome through its output queue.

    Args:
        messages: message queue to get application requests from.
        outputs: message queue to post Akida processing results to.
    """

    model = None
    stop_requested = False
    akida_power = AkidaPower()
    while not stop_requested:
        # Block till next message is received
        message = messages.get()
        akida_power.fps_measure(start_timer=True)
        # Extract command and parameters
        command = message[0]
        param = message[1]
        #print('Command:',command)
        #print('Param', param)
        if command == Command.Reset:
            # Reset the Akida model
            try:
                model = reset_model(model_path=param[0], hw_enabled=param[1])
                print("ProcessingUtils: Updated akida model to " + param[0])
                print("ProcessingUtils: Running model on " +
                      ("software" if param[1] is False else "hardware"))
            except (RuntimeError, TypeError) as err:
                print("ProcessingUtils: " + err)
            except:
                print("ProcessingUtils: Invalid model file " + param[0])
        elif command == Command.Evaluate and model is not None:
            # Classify received frame
            frame = param
            potentials = evaluate_frame(model, frame)
            akida_power.fps_measure()
            # Send back potentials and power measures (if available)
            power_stats = akida_power.get_power_stats()
            out_message = [potentials, power_stats]
            outputs.put(out_message)
        elif command == Command.Fit and model is not None:
            # Learn received frame
            frame = param[0]
            label = param[1]
            output = fit(model=model, frame=frame, label=label)
            akida_power.fps_measure()
            # Send back potentials and power measures (if available)
            power_stats = akida_power.get_power_stats()
            out_message = [output, power_stats]
            outputs.put(out_message)
        elif command == Command.Predict and model is not None:
            # Predict received frame
            frame = param[0]
            num_classes = param[1]
            #print("CLASSES: ",num_classes)
            #prediction = predict(model, frame, num_classes)
            prediction = forward_frame(model, frame)
            #print('PREDICTION',prediction)
            akida_power.fps_measure()
            # Send back prediction and power measures (if available)
            power_stats = akida_power.get_power_stats()
            out_message = [prediction, power_stats]
            outputs.put(out_message)
        elif command == Command.Stop:
            # Stop processing
            stop_requested = True
        else:
            print("ProcessingUtils: Unsupported message type " + str(command))
    print("ProcessingUtils: Exiting worker process")
