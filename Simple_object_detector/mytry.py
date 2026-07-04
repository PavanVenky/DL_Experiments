import numpy as np
import keras


def build_model(input_shape=(224,224,3),classes_cnt=2):
    base_model = keras.applications.vgg16.VGG16(input_shape=input_shape,include_top=False,weights='imagenet')
    
    for layer in base_model.layers:
        layer.trainable = False
    
    inputs = keras.Input(shape=input_shape)

    x = base_model(inputs,training=False)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(512,activation='relu')(x)
    x = keras.layers.Dense(256,activation='relu')(x)

    #classification head
    class_head = keras.layers.Dense(128,activation='relu')(x)
    class_head = keras.layers.Dense(classes_cnt,activation='softmax')(class_head)
    
    #detection head
    det_head = keras.layers.Dense(128,activation='relu')(x)
    det_head = keras.layers.Dense(4,activation='linear')(det_head)

    #final model
    final_model = keras.Model(inputs=inputs,outputs=[class_head,det_head])

    final_model.summary()




build_model()