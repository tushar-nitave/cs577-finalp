"""
Course: Deep Learning CS577
Task: CRNN model

"""
import xml.etree.ElementTree as ET
import pandas as pd
import os
import shutil

from keras import layers
from keras import backend as K
from keras.layers.recurrent import LSTM
from keras.layers import  Conv2D, MaxPool2D
from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda, BatchNormalization
from keras.layers.merge import add
from keras.models import  Model
from PIL import Image
from keras.preprocessing.image import  ImageDataGenerator

import cv2
import numpy as np

import pathlib


chars = "adefghjknqrstwABCDEFGHIJKLMNOPZ0123456789"
letters = [letter for letter in chars]
num_classes = len(letters)+1
max_text_len = 9


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def get_model():
    """
    Building CRNN architecture
    :return: model
    """

    inputs = Input(name="the_input", shape=(64, 64, 3), dtype='float32')
    print("1")
    inner = Conv2D(64, (3,3), padding='same', name='conv1', kernel_initializer='he_normal')(inputs)
    print("2")
    inner = BatchNormalization()(inner)
    print("3")
    inner = Activation('relu')(inner)
    print("4")
    inner = MaxPool2D(pool_size=(2,2), name='max1')(inner)
    print("5")
    inner = Reshape(target_shape=((32, 2048)), name="reshape")(inner)
    print("6")
    lstm_1 = LSTM(8, return_sequences=True, kernel_initializer='he_normal', name="lstm1")(inner)
    print("7")
    lstm_1b = LSTM(8, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm1_b')(inner)
    print("8")
    reversed_lstm_1b = Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1))(lstm_1b)
    print("9")
    lstm_1_merged = add([lstm_1, reversed_lstm_1b])
    print("10")
    lstm_1_merged = BatchNormalization()(lstm_1_merged)
    print("11")

    inner = Dense(num_classes, kernel_initializer='he_normal', name="dense2")(lstm_1_merged)
    print("12")
    y_pred = Activation('softmax', name="softmax")(inner)
    print("13")
    labels = Input(name="the_labels", shape=[9], dtype='float32')
    print("14")
    input_length = Input(name="input_length", shape=[1], dtype="int64")
    print("15")
    label_length = Input(name="label_length", shape=[1], dtype="int64")
    print("16")
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name="ctc")([y_pred, labels, input_length, label_length])
    print("17")
    return Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)


if __name__ == "__main__":

    # extract labels from xml file
    data = ET.parse("../data/word_train/word.xml")

    labels = []

    for child in data.getroot():
        labels.append(child.attrib['tag'])

    labels = pd.DataFrame(labels)
    labels.to_csv("../data/data.csv")
    # print(labels)

    # get images from all the directories

    # os.mkdir("../data/train/")
    path = "../data/word_train/word"
    for directory in os.listdir(path):
        for image in os.listdir(os.path.join(path, directory)):
            src = os.path.join(path, directory)
            shutil.copy(os.path.join(src, image), "../data/train/")

    #  resize all the train images
    for image in os.listdir("../data/train/"):
        img = cv2.imread(os.path.join("../data/train", image))
        img = cv2.resize(img, (64, 64))
        cv2.imwrite(image, img)


    train_data = []

    for image in os.listdir("../data/train"):
        path = os.path.join("../data/train", image)
        img = Image.open(path)
        img = img.resize((64, 64))
        train_data.append(np.array(img))

    train_data = np.array(train_data)
    print(train_data.shape)
    model = get_model()

    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer="adam")

    model.fit(train_data[:1], np.array(labels), epochs=20)

