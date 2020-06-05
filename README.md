# Keyword Detection at Edge
<br>
A deep learning model network using depthwise separable convolution network for edge devices for keyword speech recognition as part of the 6 Semester Mini project.
<br>
Dataset : https://www.kaggle.com/c/tensorflow-speech-recognition-challenge
<br>

## Desktop Environment 
To build and test in Desktop Enviroment : 
``` 
   pip install requirements_desktop.txt
   jupyter-notebook Keyword_Detection_at_Edge.ipynb
```
The model built is saved in the folder - model

## Edge Devices
To run the keyword detection model in edge devices: 

Python == 3.7
Architecture : ARM32

```
  pip install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_armv7l.whl
  pip install requirements_micro.txt
  python script.py
```
