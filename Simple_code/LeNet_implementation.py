import cv2
import keras
from keras import layers,losses,metrics
from keras.datasets import mnist
from keras import utils
import numpy as np
import matplotlib as plt
import onnx
import tf2onnx

#https://paravisionlab.co.in/lenet5-keras/

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#set dataset
x_train_main = x_train[:50000]
y_train_main = y_train[:50000]
x_eval = x_train[50000:]
y_eval = y_train[50000:]

#normalize data to [0 1]
x_train_main = x_train_main/255.0
x_eval = x_eval/255.0
x_test = x_test/255.0

#reshape data
x_train_main = x_train_main.reshape(50000,28,28,1)
x_eval = x_eval.reshape(10000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)

#convert to categorical vectors
y_train_main = keras.utils.to_categorical(y_train_main,10)
y_eval = keras.utils.to_categorical(y_eval,10)
y_test = keras.utils.to_categorical(y_test,10)

#Build LeNet Model
model = keras.Sequential()

model.add(layers.InputLayer(input_shape=(28,28,1)))
model.add(layers.Conv2D(filters=6,kernel_size=(5,5),activation='tanh'))
model.add(layers.AveragePooling2D(pool_size=(2,2)))
model.add(layers.Conv2D(filters=16,kernel_size=(5,5),activation='tanh'))
model.add(layers.AveragePooling2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(units=120,activation='tanh'))
model.add(layers.Dense(units=84,activation='tanh'))
model.add(layers.Dense(units=10,activation='softmax'))

# #convert to onnx model
# onnx_model,_=tf2onnx.convert.from_keras(model,opset=13)
# with open("sample.onnx","wb") as f:
#     f.write(onnx_model.SerializeToString())

model.summary()
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train_main,y_train_main,batch_size=20,epochs=10,validation_data=(x_eval,y_eval))

test_loss,test_acuracy = model.evaluate(x_test,y_test)
print("test_accuracy:", test_acuracy)


