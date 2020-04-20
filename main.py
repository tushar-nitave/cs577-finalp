"""
@author: Tushar Nitave (https://www.github.com/tushar-nitave)
        Jasmeet Narang ()

Course: Deep Learning CS577 Spring 20

Task: CRNN model for Scene Text Recognition

Date: April 26 2020
"""

import pandas as pd
import os
import numpy as np
import random

from keras import layers
from keras import backend as K
from keras.layers.recurrent import LSTM
from keras.layers import  Conv2D, MaxPooling2D, concatenate
from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda, BatchNormalization
from keras.layers.merge import add
from keras.models import  Model
from tensorflow import keras

import cv2
import matplotlib.pyplot as plt

from parameter import *


def labels_to_text(labels):
    return ''.join(list(map(lambda x: letters[int(x)], labels)))


def text_to_labels(text):
    return list(map(lambda x: letters.index(x), text))


class TextImageGenerator:

    def __init__(self, img_dirpath, img_w, img_h, batch_size, downsample_factor, max_len=9):

        self.img_h = img_h
        self.img_w = img_w
        self.batch_size = batch_size
        self.max_len = max_len
        self.downsample_factor = downsample_factor
        self.img_dirpath = img_dirpath
        self.img_dir = os.listdir(self.img_dirpath)
        self.n = len(self.img_dir)
        self.indexes = list(range(self.n))
        self.cur_index = 0
        self.imgs = np.zeros((self.n, self.img_h, self.img_w))
        self.texts = []

    def build_data(self):
        dir = "../data"
        print("\nBuilding data started..")

        if str(os.path.basename(os.path.normpath(self.img_dirpath))) == "train":
            text_data = pd.read_csv(os.path.join(dir, "train_label.csv"), header=None)
        if str(os.path.basename(os.path.normpath(self.img_dirpath))) == "val":
            text_data = pd.read_csv(os.path.join(dir, "val_label.csv"), header=None)

        for i, img_file in enumerate(self.img_dir):
            img = cv2.imread(os.path.join(self.img_dirpath, img_file), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (self.img_w, self.img_h))
            img = img.astype(np.float32)
            img = (img/255.0) * 2 - 1  # why ?

            self.imgs[i, :, :] = img
            self.texts.append(text_data.iloc[i:i+1,0:1].values[0][0])
        print(len(self.texts)==self.n)
        print("\nBuild finished")

    def next_sample(self):
        self.cur_index += 1
        if self.cur_index > self.n:
            self.cur_index = 0
            random.shuffle(self.indexes)
        return self.imgs[self.indexes[self.cur_index]], self.texts[self.indexes[self.cur_index]]

    def next_batch(self):
        while True:
            x_data = np.ones([self.batch_size, self.img_w, self.img_h, 1])
            y_data = np.ones([self.batch_size, self.max_len])
            input_length = np.ones((self.batch_size, 1)) * (self.img_w // self.downsample_factor - 2)
            label_length = np.zeros((self.batch_size, 1))

            for i in range(self.batch_size):
                img, text = self.next_sample()
                img = img.T
                img = np.expand_dims(img, -1)
                x_data[i] = img
                text = text[:9]
                y_data[i] = text_to_labels(text.zfill(self.max_len))
                label_length[i] = len(text)

            inputs = {

                    'the_input': x_data,
                    'the_labels': y_data,
                    'input_length': input_length,
                    'label_length': label_length
            }

            outputs = {'ctc': np.zeros([self.batch_size])}

            yield (inputs, outputs)


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def get_model(training):
    input_shape = (img_w, img_h, 1)  # (128, 64, 1)

    # Make Network
    inputs = Input(name='the_input', shape=input_shape, dtype='float32')  # (None, 128, 64, 1)

    # Convolution layer (VGG)
    inner = Conv2D(64, (3, 3), padding='same', name='conv1', kernel_initializer='he_normal')(
        inputs)  # (None, 128, 64, 64)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = MaxPooling2D(pool_size=(2, 2), name='max1')(inner)  # (None,64, 32, 64)

    inner = Conv2D(128, (3, 3), padding='same', name='conv2', kernel_initializer='he_normal')(
        inner)  # (None, 64, 32, 128)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = MaxPooling2D(pool_size=(2, 2), name='max2')(inner)  # (None, 32, 16, 128)

    inner = Conv2D(256, (3, 3), padding='same', name='conv3', kernel_initializer='he_normal')(
        inner)  # (None, 32, 16, 256)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = Conv2D(256, (3, 3), padding='same', name='conv4', kernel_initializer='he_normal')(
        inner)  # (None, 32, 16, 256)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = MaxPooling2D(pool_size=(1, 2), name='max3')(inner)  # (None, 32, 8, 256)

    inner = Conv2D(512, (3, 3), padding='same', name='conv5', kernel_initializer='he_normal')(
        inner)  # (None, 32, 8, 512)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = Conv2D(512, (3, 3), padding='same', name='conv6')(inner)  # (None, 32, 8, 512)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = MaxPooling2D(pool_size=(1, 2), name='max4')(inner)  # (None, 32, 4, 512)

    inner = Conv2D(512, (2, 2), padding='same', kernel_initializer='he_normal', name='con7')(
        inner)  # (None, 32, 4, 512)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)

    # CNN to RNN
    inner = Reshape(target_shape=((32, 2048)), name='reshape')(inner)  # (None, 32, 2048)
    inner = Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(inner)  # (None, 32, 64)

    # RNN layer
    lstm_1 = LSTM(256, return_sequences=True, kernel_initializer='he_normal', name='lstm1')(inner)  # (None, 32, 512)
    lstm_1b = LSTM(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm1_b')(inner)
    reversed_lstm_1b = Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1))(lstm_1b)

    lstm1_merged = add([lstm_1, reversed_lstm_1b])  # (None, 32, 512)
    lstm1_merged = BatchNormalization()(lstm1_merged)

    lstm_2 = LSTM(256, return_sequences=True, kernel_initializer='he_normal', name='lstm2')(lstm1_merged)
    lstm_2b = LSTM(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm2_b')(
        lstm1_merged)
    reversed_lstm_2b = Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1))(lstm_2b)

    lstm2_merged = concatenate([lstm_2, reversed_lstm_2b])  # (None, 32, 1024)
    lstm2_merged = BatchNormalization()(lstm2_merged)

    # transforms RNN output to character activations:
    inner = Dense(num_classes, kernel_initializer='he_normal', name='dense2')(lstm2_merged)  # (None, 32, 63)
    y_pred = Activation('softmax', name='softmax')(inner)

    labels = Input(name='the_labels', shape=[max_text_len], dtype='float32')  # (None ,8)
    input_length = Input(name='input_length', shape=[1], dtype='int64')  # (None, 1)
    label_length = Input(name='label_length', shape=[1], dtype='int64')  # (None, 1)

    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
        [y_pred, labels, input_length, label_length])  # (None, 1)

    if training:
        return Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)


if __name__ == "__main__":

    model = get_model(training=True)

    # prepare train data
    dir_path = "../data/train/"
    train = TextImageGenerator(dir_path, img_w, img_h, batch_size, downsample_factor)
    train.build_data()

    # prepare validation data
    dir_path = "../data/val/"
    val = TextImageGenerator(dir_path, img_w, img_h, val_batch_size, downsample_factor)
    val.build_data()

    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer="adam")
    history = model.fit_generator(generator=train.next_batch(),
                        steps_per_epoch=int(train.n/batch_size),
                        epochs=1,
                        validation_data=train.next_batch(),
                        validation_steps=int(val.n/val_batch_size))

    history = history.history

    epochs = range(1, len(history['loss'])+1)
    plt.show(epochs, history['loss'])
