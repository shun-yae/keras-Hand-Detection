import os
import glob
import numpy as np
import cv2

from keras.models import Sequential, Model
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Dropout, Convolution2D, Flatten, MaxPooling2D, GlobalAveragePooling2D, Input
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adagrad, Adam, SGD
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# parameter
data_dir = "hand"
W, H = 100,100
epoch = 10

input_dir = glob.glob(data_dir + "/*")
learn_class = len(os.listdir(data_dir))


def data_create():
    """ Read Images """

    label = []
    datas = []

    for i in input_dir:
        L = i[-1:]
        files = glob.glob(i + "/*")

        for j in files:
            label.append(L)
            img = cv2.imread(j)
            img = cv2.resize(img, (W, H))
            datas.append(img / 255.0)

    datas = np.array(datas, dtype="float16")
    labels = to_categorical(label)

    return datas, labels

data = data_create()
# model Set Up
model_vgg = VGG16(
    include_top=False, weights="imagenet",
    input_tensor=Input(shape=(W, H, 3)))

set_model = model_vgg.output
set_model = GlobalAveragePooling2D()(set_model)
set_model = Dense(1024, activation="relu")(set_model)
set_model = Dense(512, activation="relu")(set_model)
pred = Dense(learn_class, activation="softmax")(set_model)
model = Model(input=model_vgg.input, output=pred)

for i in model_vgg.layers[:15]:
    i.trainable = False

sgd = SGD(lr=0.0001, momentum=0.9)
loss = "categorical_crossentropy"

model.compile(optimizer=sgd, loss=loss, metrics=["accuracy"])
model.summary()

early = EarlyStopping(monitor="val_loss", patience=3)

model.fit(
    data[0], data[1], epochs=epoch, batch_size=25,
    validation_split=0.15, callbacks=[early])

# create model file
model_json = model.to_json()
open("./hand_detection.json", "w").write(model_json)
model.save_weights("./hand_detection.h5")
