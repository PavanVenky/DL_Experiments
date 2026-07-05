import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import keras
from keras.layers import Conv2D,MaxPooling2D,Flatten,AveragePooling2D,Activation, Input,Dense,InputLayer
from keras.models import Sequential,Model
from keras.losses import CategoricalCrossentropy
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report

print(cv2.__version__)

path = "Datasets\Cricket_Legends_data"

folders_path = os.path.join(path,os.listdir(path)[0])

print(folders_path)
classes = os.listdir(folders_path)

labels = []

# LABELS DISPLAY AND ANALYSIS
# for player in classes:
#     player_dir = os.path.join(folders_path,player)
#     player_img_cnt = len(os.listdir(player_dir))
#     labels.extend([player]*player_img_cnt)

# classes, counts = np.unique(labels, return_counts=True)
# plt.figure(figsize=(10, 6))
# bars = plt.barh(classes, counts, color='skyblue')
# plt.xlabel('Total image count')
# plt.ylabel('Class name')
# plt.title('Images per player class')

# for bar, count in zip(bars, counts):
#     plt.text(
#         bar.get_width() + 1,      # x position just after the bar
#         bar.get_y() + bar.get_height() / 2,
#         str(count),
#         va='center',
#         ha='left'
#     )

# plt.tight_layout()
# plt.show()

#PRE-PROCESS AND DATA SPLIT
X = []
Y = []
label_name = {}

for idx,player in enumerate(classes):
    player_dir = os.path.join(folders_path,player)
    player_img_cnt = len(os.listdir(player_dir))
    label_name[player] = idx
    for img_name in os.listdir(player_dir):
        each_player_img_path = os.path.join(player_dir,img_name)
        img = cv2.imread(each_player_img_path)
        img = cv2.resize(img,(128,128))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        X.append(img)
        Y.append(idx)

#https://www.kaggle.com/code/evilspirit05/lenet-cricketer-classifier-grad-cam-maximization#summary-of-our-data

#convert lists to numpy
X = np.array(X)
Y = np.array(Y)

X = X/255.0 #normalize data

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=42)
print(f"Total Images: {len(X)} Train data = {len(x_train)} test data = {len(x_test)}")

print(f"labels: {label_name}")
print(f"train data shape {x_train.shape}")
print(f"test data shape {x_test.shape}")
print(f"train labels shape {y_train.shape}")
print(f"test lables shape {y_test.shape}")

num_classes = len(classes)
input_shape = (128,128,3)
print(f"Total classes {num_classes}")

# Implement Model
model = Sequential()
model.add(Input(shape=input_shape))

model.add(Conv2D(filters=16,kernel_size=(5,5),padding='same',activation='relu', name='conv2D_1'))
model.add(AveragePooling2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=6,kernel_size=(5,5),padding='same',activation='relu', name='conv2D_3'))
model.add(AveragePooling2D(pool_size=(2,2),strides=(2,2)))

model.add(Flatten())
model.add(Dense(120))
model.add(Activation('relu'))
model.add(Dense(84))
model.add(Activation('relu'))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(learning_rate=0.001),
              metrics=['accuracy'])
print(model.summary())

# Train Model...
from keras.callbacks import EarlyStopping,LearningRateScheduler,ReduceLROnPlateau
early_stopping = EarlyStopping(monitor='val_loss',patience=3,restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                              factor=0.1, 
                              patience=2, 
                              min_lr=1e-6, 
                              verbose=1)
lr=LearningRateScheduler(lambda epoch,lr:lr*0.1 if epoch%5==0 and epoch !=0 else lr)

history = model.fit(x_train,y_train,batch_size=32,
                    epochs=50,
                    validation_data=(x_test,y_test))#,
                    #callbacks=[early_stopping,reduce_lr,lr])

# Plotting Loss and Accuracy
plt.figure(figsize=(12, 8))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

#Evaluate Model
eval = model.evaluate(x_test,y_test,verbose=1)
print(f"Test loss {eval[0]}")
print(f"Test Accuracy {eval[1]}")

#Confusion Matrix
# build ordered class names from label_name dict
class_names = [None] * len(label_name)
for name, idx in label_name.items():
    class_names[idx] = name

# predict class indices
y_pred_probs = model.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# plot
plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45, ha='right')
plt.yticks(tick_marks, class_names)

plt.ylabel('True label')
plt.xlabel('Predicted label')

# annotate counts
thresh = cm.max() / 2
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j],
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')

plt.tight_layout()
plt.show()


# Grad-CAM can work with Sequential or functional models.
def create_lenet(input_shape,num_classes,learning_rate=0.001):
    input = Input(shape=input_shape)

    x = Conv2D(filters=6,kernel_size=(5,5),padding='same', name='conv2D_1')(input)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)

    x = Conv2D(filters=16,kernel_size=(5,5),padding='same', name='conv2D_2')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)

    x = Conv2D(filters=120,kernel_size=(5,5),padding='same', name='conv2D_3')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)

    x = Flatten()(x)
    x = Dense(120,activation='relu')(x)
    x = Dense(84,activation='relu')(x)

    outputs = Dense(num_classes,activation='softmax')(x)

    model = Model(inputs=input,outputs=outputs)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=Adam(learning_rate=learning_rate),
                  metrics=['accuracy'])
    return model

# Leave the trained Sequential model in `model` and do not overwrite it here.
# If you want to create a separate functional model for experimentation, uncomment the next line.
# gradcam_model = create_lenet(input_shape,num_classes)
# gradcam_model.summary()


# Grad-CAM visualization for the conv2D_3 layer
# Make sure the layer name matches the layer you want to inspect.

def sequential_to_functional(seq_model):
    x = Input(shape=seq_model.input_shape[1:])
    y = x
    for layer in seq_model.layers:
        if isinstance(layer, InputLayer):
            continue
        y = layer(y)
    return Model(inputs=x, outputs=y)


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    import tensorflow as tf

    func_model = sequential_to_functional(model)
    grad_model = Model(
        func_model.inputs,
        [func_model.get_layer(last_conv_layer_name).output, func_model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + tf.keras.backend.epsilon())
    return heatmap.numpy()


def save_and_display_gradcam(img, heatmap, alpha=0.4):
    img_uint8 = np.uint8(255 * img)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.resize(heatmap, (img_uint8.shape[1], img_uint8.shape[0]))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    superimposed_img = cv2.addWeighted(img_uint8, 1 - alpha, heatmap, alpha, 0)
    return superimposed_img


last_conv_layer_name = 'conv2D_3'

img = x_test[0]
img_array = np.expand_dims(img, axis=0)

preds = model.predict(img_array)
pred_class = int(np.argmax(preds[0]))
print('Predicted class:', class_names[pred_class])

heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=pred_class)
superimposed_img = save_and_display_gradcam(img, heatmap)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title('Input Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(heatmap, cmap='jet')
plt.title(f'Grad-CAM heatmap ({last_conv_layer_name})')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(superimposed_img)
plt.title('Superimposed Grad-CAM')
plt.axis('off')

plt.tight_layout()
plt.show()



