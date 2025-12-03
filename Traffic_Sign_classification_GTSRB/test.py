import keras
from keras import Sequential
from keras import layers

model = keras.Sequential()
model.add(layers.Conv2D(filters=5,kernel_size=(5,5),activation='relu'))




def greeting():
    print("Hello inside docker!!..")

if __name__ == "__main__":
       greeting()