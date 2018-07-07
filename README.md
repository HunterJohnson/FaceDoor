# FaceDoor
door lock system using Facial Recognition on the Raspberry Pi

1. Credit for motion_detection goes to Adrian Rosebrock at PyImageSearch
https://www.pyimagesearch.com/2015/06/01/home-surveillance-and-motion-detection-with-the-raspberry-pi-python-and-opencv/

--for the facial recognition component, instead of using VGG_Face & Keras it may be easier
to use this API that I discovered after completing this project

See here: https://github.com/ageitgey/face_recognition

# HARDWARE

* Raspberry Pi 3
* Camera of choice (I used a NoIR that cost like $25 on Amazon)
* 12V lock style solenoid
* 5V relay
* 12V power supply + DC adapter

# SOFTWARE

* picamera 
* OpenCV
* Keras / Tensorflow
* random Python libraries
