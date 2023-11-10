from os.path import basename, dirname, splitext, join, isfile
from pathlib import Path
from urllib.parse import urlparse
from tempfile import TemporaryDirectory
import numpy as np
import copy

from tensorflow.keras.utils import get_file

from akida import (Model, Layer, LayerType, InputData, FullyConnected,
                   SeparableConvolutional)
from cnn2snn import load_quantized_model, convert


class ModelProvider():
    """
    Object that provides function and tools to manage an akida.Model.
    """

    # Private variables to store temporary model for edge learning
    _tmp_dir = TemporaryDirectory(prefix='edge_learning')
    _tmp_model_file = join(_tmp_dir.name, 'edge_learning.fbz')
    _akida_cache_folder = join(Path.home(), ".akida/models")

    def __init__(self, **kwargs):
        pass

    def fetch_model(self, model_url, input_is_image=False):
        """
        Downloads a file from a URL if it is not already in the cache.

        :args:
            model_url(str): Original URL of the file.
            input_is_image(bool): True if input is an image (3-D 8-bit
            input with 1 or 3 channels) followed by QuantizedConv2D. Akida model
            input will be InputConvolutional. If False, Akida model input will
            be InputData.
        """

        # Check akida cache folder to get an existing converted keras model
        filename = model_url.rsplit('/', 1)[-1].replace(".h5", ".fbz")
        filepath = join(self._akida_cache_folder, filename)
        if isfile(filepath):
            return filepath

        # Parse URL
        model_path = model_url
        parsed_url = urlparse(model_url)
        # Check if it's a local file
        if bool(parsed_url.scheme):
            # Downloads file
            model_path = self._get_model(model_url=model_url)

        # Check file extension. If it's a .h5, it needs to be converted to akida
        if model_path.endswith(".h5"):
            # Converts .h5 file to .fbz
            model_path = self._convert_to_akida(model_path=model_path,
                                                input_is_image=input_is_image)

        return model_path

    def prepare_edge_learning(self, model_path, num_classes, num_neurons,
                              num_weights):
        """
        Prepares model for edge learning.

        :args:
            model_path(str): the path to the original model
            num_classes(int): number of maximum classes
            num_neurons(int): number of neurons per class
            num_weights(int): number of weights
        """
        tmp_model = Model(model_path)

        tmp_model.pop_layer()

        # Check last layer output_bits.
        # If output bits != 1, the following code adds an identity layer
        # to get a compatible output_bits value.
        last_layer = tmp_model.get_layer(tmp_model.get_layer_count() - 1)
        if last_layer.parameters.act_bits != 1:
            self._add_identity_layer(model=tmp_model)

        layer_fc = FullyConnected(name='edge_layer',
                                  units=num_classes * num_neurons,
                                  activation=False)

        tmp_model.add(layer_fc)
        tmp_model.compile(num_weights=num_weights,
                          num_classes=num_classes,
                          learning_competition=0.1)

        tmp_model.save(self._tmp_model_file)
        del tmp_model
        return self._tmp_model_file

    def _add_identity_layer(self, model):
        last_layer = model.get_layer(model.get_layer_count() - 1)
        ident_params, ident_dw_weights, ident_pw_weights = self._get_weights_params_identity(
            last_layer)
        identity_layer = Layer(ident_params, f"{last_layer.name}_identity")
        model.add(identity_layer)
        identity_layer.set_variable("weights", ident_dw_weights)
        identity_layer.set_variable("weights_pw", ident_pw_weights)

    def _get_weights_params_identity(self, layer):
        """
        Creates an 'identity' convolutional layer parameters and its weights.
        """
        out_dims = layer.output_dims
        nb_chan = out_dims[2]
        dw_weights = np.zeros((3, 3, nb_chan, 1), dtype=np.int8)
        pw_weights = np.zeros((1, 1, nb_chan, nb_chan), dtype=np.int8)
        for i in range(nb_chan):
            dw_weights[1, 1, i, 0] = 1
            pw_weights[0, 0, i, i] = 1

        # create a layer to have default parameters
        identity_layer = SeparableConvolutional(name=f"{layer.name}_pooling",
                                                kernel_size=(3, 3),
                                                filters=nb_chan,
                                                threshold=0,
                                                act_bits=1)
        return copy.copy(identity_layer.parameters), dw_weights, pw_weights

    def _get_model(self, model_url):
        # Get model filename
        model_filename = basename(urlparse(model_url).path)
        # Fetch model file from url
        return get_file(fname=model_filename,
                        origin=model_url,
                        cache_subdir='models')

    def _convert_to_akida(self, model_path, input_is_image):
        # Load quantized model
        keras_model = load_quantized_model(model_path)
        # Extract path and filename w/o extension
        model_folder = dirname(model_path)
        filename = splitext(basename(model_path))[0]
        # Convert model for akida
        model_path = join(model_folder, filename) + ".fbz"
        convert(keras_model,
                file_path=model_path,
                input_is_image=input_is_image)

        return model_path
