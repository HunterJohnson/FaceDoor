
import io
import picamera
import cv2
import random as rand
import numpy
import time

def face_detected():
    found = False
    start = time.time()
    end = 0
    diff = 0
    while(found != True or diff <= 30.0): # function has 30 seconds to find a face, otherwise revert to motion_detection()
        #Create a memory stream so photos doesn't need to be saved in a file
        stream = io.BytesIO()

        #Get the picture (low resolution, so it should be quite fast)
        #Here you can also specify other parameters (e.g.:rotate the image)
        with picamera.PiCamera() as camera:
            camera.resolution = (320, 240)
            camera.capture(stream, format='jpeg')

    #Convert the picture into a numpy array
        buff = numpy.fromstring(stream.getvalue(), dtype=numpy.uint8)

    #create an OpenCV image
        image = cv2.imdecode(buff, 1)

    #Load a cascade file for detecting faces
        face_cascade = cv2.CascadeClassifier('/FaceDoor/haarcascade_frontalface_default_2.xml')

    #Convert to grayscale
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    #Look for faces in the image using the loaded cascade file
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        print("Found "+str(len(faces))+" face(s)")
        image_faces = []
    #Draw a rectangle around every found face
        for (x,y,w,h) in faces:
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
            image_faces.append(image[y:(y+h),x:(x+w)]) # this line of code crops the image to keep only the bounded face

        if(len(faces) >= 1):
            #save result image
            x = rand.randint(1000,1200) # should probably use a timestamp to label images instead
            s = str(x)
            for i, face in enumerate(image_faces):
                cv2.imwrite("face-" + s + ".jpg", face)
            found = True
            camera.close()
            return True
        end = time.time()
        diff = end - start
        
    return False
