#Verifying faces found in AR_Face / LFW dataset, this file is used to train our model 

# once we have a .h5 file with the architecture + weights
# of the trained VGG_Face model, we use that to make predictions on new images

import os
import time as t
import numpy as np
import pandas as pd
import traceback
import logging

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from keras_vggface.vggface import VGGFace

from keras.models import Model
from keras.layers import Dense, Dropout
from keras.optimizers import Adamax
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator, array_to_img

def preprocess_input_vgg(x):
    """Wrapper around keras.applications.vgg16.preprocess_input()
    to make it compatible for use with keras.preprocessing.image.ImageDataGenerator's
    `preprocessing_function` argument.
    
    Parameters
    ----------
    x : a numpy 3darray (a single image to be preprocessed)
    
    Note we cannot pass keras.applications.vgg16.preprocess_input()
    directly to to keras.preprocessing.image.ImageDataGenerator's
    `preprocessing_function` argument because the former expects a
    4D tensor whereas the latter expects a 3D tensor. Hence the
    existence of this wrapper.
    
    Returns a numpy 3darray (the preprocessed image).
    
    """
    from keras.applications.vgg16 import preprocess_input
    X = np.expand_dims(x, axis=0)
    X = preprocess_input(X)
    return X[0]

def save_prediction_examples(num_examples=1, validation_image_generator=None, model=None, folder_path='results/', model_name='name'):
    """Helper method to save examples of images and prediction probabilities"""
    X_val_sample, _ = next(validation_image_generator)
    y_pred = model.predict(X_val_sample)
    for idx, x, y in zip(range(num_examples), X_val_sample[:num_examples], y_pred.flatten()[:num_examples]):
        s = pd.Series(y)
        axes = s.plot(kind='bar')
        axes.set_xlabel('Class')
        axes.set_ylabel('Probability')
        axes.set_ylim([0, 1])

        try:
            np.save(os.path.join(folder_path, model_name + "_result_guess_" + str(idx) + ".npy"), y)
        except Exception as e:
            logging.error(traceback.format_exec())
        try:
            np.save(os.path.join(folder_path, model_name + "_prediction_sample_" + str(idx) + ".npy"), x)
        except Exception as e:
            logging.error(traceback.format_exec())
        try:
            plt.savefig(os.path.join(folder_path, model_name + "_result_graph_" + str(idx) + ".png"))
        except Exception as e:
            logging.error(traceback.format_exec())
        try:
            plt.imsave(os.path.join(folder_path, model_name + "_prediction_sample_" + str(idx) + ".png"), x)
        except Exception as e:
            logging.error(traceback.format_exec())

#create constants
NUM_CLASSES = 136 #there are 136 unique individuals in the AR_Face dataset: 76 men, 60 women
NUM_TRAIN_IMAGES = 2021
NUM_VAL_IMAGES = 673
EPOCHS = 50
TRAINING_BATCH_SIZE = 8
STEPS_PER_EPOCH = round(NUM_TRAIN_IMAGES / TRAINING_BATCH_SIZE)
VALIDATION_BATCH_SIZE = 8
VALIDATION_STEPS_PER_EPOCH = round(NUM_VAL_IMAGES / VALIDATION_BATCH_SIZE)
#dump full model to disk every 10 epochs
MODEL_SAVER_PERIOD = EPOCHS / 10

#create training and validation data generators
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input_vgg,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
train_generator = train_datagen.flow_from_directory(directory='../data/split_sets/training',
                                                    target_size=[224, 224],
                                                    batch_size=TRAINING_BATCH_SIZE,
                                                    class_mode='categorical')

validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input_vgg)
validation_generator = validation_datagen.flow_from_directory(directory='../data/split_sets/validation',
                                                              target_size=[224, 224],
                                                              batch_size=VALIDATION_BATCH_SIZE,
                                                              class_mode='categorical')
vggface = VGGFace(weights='vggface', input_shape=(224, 224, 3))


#fc7/relu is the layer right before the predictions in vggface
final_vggface_layer = vggface.get_layer('fc7/relu').output
vggface_prediction = Dropout(0.2)(final_vggface_layer)
vggface_prediction = Dense(units=NUM_CLASSES, activation='softmax', name='predictions')(vggface_prediction)
vggface_model = Model(inputs=vggface.input, outputs=vggface_prediction)


timestamp = "results/" + t.strftime("%Y%m%d-%H%M%S")
if not os.path.exists(timestamp):
    os.makedirs(timestamp)

vggface_csv_logger = CSVLogger(timestamp + '/vggface_adamax_d2_training_log.csv', append=True, separator=',')
vggface_model_saver = ModelCheckpoint(timestamp + '/vggface.weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=0, save_best_only=True, mode='auto', period=MODEL_SAVER_PERIOD)


#freeze feature layers, train top layers

for layer in vggface_model.layers:
    if layer.name in ['fc6', 'fc6/relu', 'fc7', 'fc7/relu', 'predictions']:
        continue
    layer.trainable = False

#using adamax optimizer
adamax = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

#compile and train model


vggface_model.compile(adamax, 'categorical_crossentropy', metrics=['accuracy'])
vggface_model.fit_generator(train_generator,
                    STEPS_PER_EPOCH,
                    epochs=EPOCHS,
                    validation_data=validation_generator,
                    validation_steps=VALIDATION_STEPS_PER_EPOCH,
                    callbacks=[vggface_csv_logger, vggface_model_saver]);

#saveexample predictions with images for each model from the validation generator

save_prediction_examples(num_examples=4, validation_image_generator=validation_generator, model=vggface_model, folder_path=timestamp, model_name='vggface_d2_adamax')
