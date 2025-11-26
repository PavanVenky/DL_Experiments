import keras
from keras import layers
import cv2
import numpy as np

img = cv2.imread("panda.jpg")
height,width,channels = img.shape
print('Height: {}, Width: {}, channels: {}'.format(height,width,channels))

model = keras.Sequential()
model.add(keras.Input(shape=(img.shape)))
model.add(keras.layers.Dense(64))
model.add(keras.layers.Dense(32))
model.add(keras.layers.Dense(2))

preproc_img = np.array([img])
result = model(preproc_img)
print(result)