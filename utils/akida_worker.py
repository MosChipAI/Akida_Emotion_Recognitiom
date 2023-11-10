from multiprocessing import Process, Queue

from utils.processing_utils import Command, worker_task


class AkidaWorker():
    """
    Class that handle worker task and messages exchanged between them.
    """

    def __init__(self, **kwargs):
        self._mode = Command.Predict
        self._do_learning = False
        self._num_classes = 1000
        

        # Create the two message queues
        self._messages = Queue(maxsize=1)
        self._outputs = Queue()

    def start(self):
        """
        Start worker task.
        """
        # Create worker process
        self._worker = Process(target=worker_task,
                               args=(self._messages, self._outputs))
        # Start worker
        self._worker.start()

    def stop(self):
        """
        Stop worker task.
        """
        # Push stop signal (will block if queue is full)
        self._messages.put([Command.Stop, None])
        print("AkidaWorker: Sent stop message to worker process.")
        # Wait for the worker to finish (1s timeout)
        self._worker.join(timeout=1)
        # Terminate worker
        self._worker.terminate()

    def set_model(self, model_path, enable_hw=False):
        """
        Set model.

        :args:
            model_path(str): model file path
            enable_hw(bool): enable/disable HW device (Default: False)
        """
        # Send reset command with model name as parameter
        print("AkidaWorker: Setting akida model to " + model_path)
        message = [model_path, enable_hw]
        self._messages.put([Command.Reset, message])

    def set_learning(self, do_learning, learned_class_id):
        """
        Set worker in learn mode.
        This mode activates learning for next shot/image.

        :args:
            do_learning(bool): enable/disable learning
            learned_class_id(int): set learned class id
        """
        self._do_learning = do_learning
        self._class_id = learned_class_id

    def set_predict(self, num_classes):
        """
        Change worker mode to 'Command.Predict'

        :args:
            num_classes(int): number of classes
        """
        self._mode = Command.Predict
        self._num_classes = num_classes

    def fetch_data(self, data):
        """Retrieves a data and posts a processing request to the
           `worker_task`.

           :args:
                data(Numpy.array): input data for akida inference.
        """
        if not self._messages.full():
            # Push Command and data to the processing queue
            if self._do_learning:
                message = [data, self._class_id]
                self._messages.put_nowait([Command.Fit, message])
                self._do_learning = False
            elif self._mode == Command.Evaluate:
                message = data
                self._messages.put_nowait([self._mode, message])
            elif self._mode == Command.Predict:
                message = [data, self._num_classes]
                self._messages.put_nowait([self._mode, message])
            else:
                raise Exception("AkidaWorker: Unexpected mode...")

    def get_output(self, block=False):
        """
        Get result from worker task.
        Worker task returns an array with:
         * potentials or prediction
         * HW power statistics (Empty if no HW available)

        :args:
            block(bool): True, will block application await results.
        """
        return self._outputs.get(block=block)
