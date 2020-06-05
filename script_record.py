import numpy as np
import tflite_runtime.interpreter as tflite
import python_speech_features
import sounddevice as sd
from scipy.io import wavfile


a = ['left', 'no', 'off', 'on', 'unknown', 'yes']

choice  = int(input("Enter 1 to record : "))

if choice == 1:
    audio = sd.rec(2*16000,samplerate=16000, channels=1)
    sd.wait()
    audio = audio[8000:24000]
    
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
    print(a[output_data.argmax(axis=-1)[0]])
    
