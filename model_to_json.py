# convert keras model to json format

from keras.models import Model, Sequential, model_from_yaml, model_from_json
import numpy as np

from keras import applications
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.layers import Dropout, Flatten, Dense, Input
from keras.layers.pooling import AveragePooling2D

def create_model():
	input_tensor = Input(shape = (img_height,img_width,3))
	base_model = applications.resnet50.ResNet50(weights=None, include_top=False,input_tensor=input_tensor)
	top_model = Sequential()
	top_model.add(Dense(1024, activation='relu',input_shape=base_model.output_shape[1:]))
	top_model.add(Dropout(0.5))
	top_model.add(Flatten())
	top_model.add(Dense(256, activation='relu'))
	top_model.add(Dropout(0.5))
	top_model.add(Dense(62, activation='softmax'))
	model = Model(input = base_model.input, output = top_model(base_model.output))
	return model


def load_trained_model(weights_path): # path to .h5 / .hdf5
   model = create_model()
   model.load_weights(weights_path)
   return model 

xmod = load_trained_model("trained_weights.h5") # or VGGFace default weights

model_json = xmod.to_json()
with open("model.json","w") as fo:
        fo.write(model_json)
        fo.close()
