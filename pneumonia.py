# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tensorflow as tf
import keras

from keras.applications.xception import Xception, preprocess_input
from keras.layers import Input, Dense, Flatten, Lambda
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator as IDG
import numpy as np
from glob import glob

IMAGE_SIZE = [299, 299]
train_path = 'chest_xray/train'
test_path = 'chest_xray/test'
val_path = 'chest_xray/val'

xception = Xception(include_top=False, input_shape=IMAGE_SIZE+[3], classifier_activation='softmax')

for layers in xception.layers:
    layers.trainable = False

folders = glob('chest_xray/train/*')
print(folders)

x = Flatten()(xception.output)

prediction = Dense(len(folders), activation='relu')(x)

model = Model(inputs=xception.input,outputs=prediction)

model.summary()

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

train_datagen = IDG(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = IDG(rescale = 1./255)

val_datagen = IDG(rescale = 1./255)

training_set = train_datagen.flow_from_directory(train_path,
                                                 target_size = (299, 299),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory(test_path,
                                            target_size = (299, 299),
                                            batch_size = 32,
                                            class_mode = 'categorical')

val_set = val_datagen.flow_from_directory(val_path,
                                            target_size = (299, 299),
                                            batch_size = None,
                                            class_mode = 'categorical')

r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=5,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.xception import preprocess_input
from keras.applications.xception import decode_predictions

image = load_img(path='chest_xray/val/PNEUMONIA/person1954_bacteria_4886.jpeg', target_size=(299, 299))
image = img_to_array(image)
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
image = preprocess_input(image)
print(model.predict(image))

import tensorflow as tf

from keras.models import load_model

model.save('pneumonia_xception.h5')

import matplotlib.pyplot as plt
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()

# accuracies
plt.plot(r.history['accuracy'], label='train accuracy')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
