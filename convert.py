

#### Invoking Akida #####
from tensorflow.keras.models import load_model
import os
import akida
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
devices = akida.devices()
#model_quantized = load_quantized_model('quantized_bkup.h5')

#emotion_dict = {0: "Angry",  1: "Happy", 2: "Neutral", 3: "Sad"}
emotion_dict = {0: "indu", 1:"nida",2:"sudhansu"}
#'''
from cnn2snn import check_model_compatibility, load_quantized_model
model = load_quantized_model('./quantized_models/akida10marv2q_ok.h5')
print("Model compatible for Akida conversion:",
      check_model_compatibility(model))
#exit(0)
from cnn2snn import quantize
model_quantized = quantize(model,
                           input_weight_quantization=8,
                           weight_quantization=4,
                           activ_quantization=4)

from cnn2snn import convert
model_akida = convert(model_quantized)
model_akida.save('Akida_9.h5')
model_akida.summary()
