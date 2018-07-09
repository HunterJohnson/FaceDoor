import keras
import tensorflow as tf
import numpy as np
from keras.preprocessing import image
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
from skimage import io, transform

permitted = ["Person_1", "Person_2"] # permitted entrant labels, change as necessary

def valid_entrant(img):
	model = VGGFace(input_shape=(224,224,3)) # (weights='vggface') for default
	model.load_weights("model_weights.h5") # custom trained
	im = image.img_to_array(img)
	im2 = transform.resize(im, (224,224,3)) #resize input image
	preds = model.predict(im2)
	x = (preds[0][0])  # class label of our most likely prediction
	if x in permitted:
		return True
	else:
		return False

