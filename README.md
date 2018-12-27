# FaceDoor

![Alt text](examples/result882.jpg?raw=true "Title")

door lock system using Facial Recognition on the Raspberry Pi

1. Credit for motion_detection goes to Adrian Rosebrock at PyImageSearch
https://www.pyimagesearch.com/2015/06/01/home-surveillance-and-motion-detection-with-the-raspberry-pi-python-and-opencv/

--> for the facial recognition component, instead of using VGG_Face + Keras it may be easier
to use this API that I discovered after completing this project

--> or you can use Keras / .h5 weights from other CNN architectures (such as DeepFace from FAIR)
https://research.fb.com/publications/deepface-closing-the-gap-to-human-level-performance-in-face-verification/

See here: https://github.com/ageitgey/face_recognition

# Hardware

* Raspberry Pi 3
* Camera of choice (I used a NoIR that cost about $25 on Amazon)
* 12V lock style solenoid
* Hammer/nails or Drill/screws to secure the lock to your door of choice (or even just strong tape if you want to test it out)
* 5V relay
* 12V power supply + DC adapter

--> Total Cost ~$50-75 not including the Pi itself

# Software

* picamera 
* OpenCV
* Keras / Tensorflow
* various Python libraries

# Project Structure

1. The project basically works as an infinite loop in app.py until the program is terminated.
   You will need to train the VGG_Face model with at least 20-30 images of each person who should be allowed access. 
   (more is even better)
   
   Then this model is saved as .h5 / .hdf5 file and used to make predictions on new input images, to determine if the
   person trying to enter is on the whitelist of permitted entrants. 
   
 ---------------------------------------------------------------------------------------------------------------------



    while(door_is_active):
    
         if motion_detected():
       
             if face_detected():
     
                 if valid_entrant(img):
          
                     unlock_door()
                 
                 else:
          
                     "Access Denied"
        
