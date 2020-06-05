import numpy as np
import tflite_runtime.interpreter as tflite
import python_speech_features
from scipy.io import wavfile


a = ['left', 'no', 'off', 'on', 'unknown', 'yes']

name = (input("Enter file: "))
rate , audio = wavfile.read(name)
audio = audio[0:16000]


X = python_speech_features.base.mfcc(audio, winlen=0.040, winstep=0.020, numcep=40, nfilt = 40, nfft = 640)

interpreter = tflite.Interpreter(model_path="converted_model.tflite")

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
X = X[np.newaxis,:,:,np.newaxis]
X = X.astype('float32')


interpreter.set_tensor(input_details[0]['index'], X)
interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])
print("Word:", a[output_data.argmax(axis=-1)[0]])
    
