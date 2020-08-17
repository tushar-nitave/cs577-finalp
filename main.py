"""
@author: Tushar Nitave (https://www.github.com/tushar-nitave)

Course: Deep Learning CS577 Spring 20

Task: CRNN model for Scene Text Recognition

Date: April 26 2020
"""

# system imports
import os
import random
import time

# helper libraries imports
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

# GPU framework imports
from keras import backend as K
from keras.layers.recurrent import LSTM
from keras.layers import Conv2D, MaxPooling2D, concatenate
from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda, BatchNormalization
from keras.layers.merge import add
from keras.models import Model
from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint

# utility imports
from utils import *




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
        """
        This method loads image data and text data for pre processing.
        Convert image to grayscale, resize to 128 x 32 and normalize.
        Clip text label length to 10 (max text len)

        :return:
        """
        dir = "data"

        if str(os.path.basename(os.path.normpath(self.img_dirpath))) == "train":
            print("\nBuilding Train data started..")
            text_data = pd.read_csv(os.path.join(dir, "train_label.csv"), header=None)
        if str(os.path.basename(os.path.normpath(self.img_dirpath))) == "val":
            print("\nBuilding Validation data started..")
            text_data = pd.read_csv(os.path.join(dir, "val_label.csv"), header=None)

        for i, img_file in enumerate(sorted(self.img_dir, key=lambda s: int(s[:-4]))):
            img = cv2.imread(os.path.join(self.img_dirpath, img_file), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (self.img_w, self.img_h))
            img = img.astype(np.float32)
            img = (img/255.0) * 2 - 1
            self.imgs[i, :, :] = img
            self.texts.append(text_data.iloc[i:i+1,0:1].values[0][0])

        print("\nBuild finished")

    def next_sample(self):
        """
        Returns image data and labels in batch for fit_generator
        :return:
        """
        self.cur_index += 1
        if self.cur_index >= self.n:
            self.cur_index = 0
            random.shuffle(self.indexes)
        return self.imgs[self.indexes[self.cur_index]], self.texts[self.indexes[self.cur_index]]

    def next_batch(self):
        """
        Prepares batch for training and validation and yields data.
        :return: inputs, outputs
        """
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
                text = str(text)
                if len(str(text)) < self.max_len:
                    y_data[i] = text_to_labels(text.ljust(self.max_len, '0'))
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

    input_shape = (img_w, img_h, 1)

    inputs = Input(name='the_input', shape=input_shape, dtype='float32')

    # Convolution layer
    inner = Conv2D(64, (3, 3), padding='same', name='conv1', kernel_initializer='he_normal')(
        inputs)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = MaxPooling2D(pool_size=(2, 2), name='max1')(inner)

    inner = Conv2D(128, (3, 3), padding='same', name='conv2', kernel_initializer='he_normal')(
        inner)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = MaxPooling2D(pool_size=(2, 2), name='max2')(inner)

    inner = Conv2D(256, (3, 3), padding='same', name='conv3', kernel_initializer='he_normal')(
        inner)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = Conv2D(256, (3, 3), padding='same', name='conv4', kernel_initializer='he_normal')(
        inner)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = MaxPooling2D(pool_size=(1, 2), name='max3')(inner)

    inner = Conv2D(512, (3, 3), padding='same', name='conv5', kernel_initializer='he_normal')(
        inner)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = Conv2D(512, (3, 3), padding='same', name='conv6')(inner)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = MaxPooling2D(pool_size=(1, 2), name='max4')(inner)

    inner = Conv2D(512, (2, 2), padding='same', kernel_initializer='he_normal', name='con7')(
        inner)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)

    # CNN to RNN
    inner = Reshape(target_shape=((32, 1024)), name='reshape')(inner)
    inner = Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(inner)  # (None, 32, 64)

    # RNN - Bi-directional LSTM
    lstm_1 = LSTM(256, return_sequences=True, kernel_initializer='he_normal', name='lstm1')(inner)  # (None, 32, 512)
    lstm_1b = LSTM(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm1_b')(inner)
    reversed_lstm_1b = Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1))(lstm_1b)

    lstm1_merged = add([lstm_1, reversed_lstm_1b])
    lstm1_merged = BatchNormalization()(lstm1_merged)

    lstm_2 = LSTM(256, return_sequences=True, kernel_initializer='he_normal', name='lstm2')(lstm1_merged)
    lstm_2b = LSTM(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm2_b')(
        lstm1_merged)
    reversed_lstm_2b = Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1))(lstm_2b)

    lstm2_merged = concatenate([lstm_2, reversed_lstm_2b])
    lstm2_merged = BatchNormalization()(lstm2_merged)

    # transforms RNN output to character activations:
    inner = Dense(num_classes, kernel_initializer='he_normal', name='dense2')(lstm2_merged)  # (None, 32, 63)
    y_pred = Activation('softmax', name='softmax')(inner)

    labels = Input(name='the_labels', shape=[max_text_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
        [y_pred, labels, input_length, label_length])

    if training:
        return Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)
    else:
        return Model(inputs=[inputs], outputs=y_pred)


if __name__ == "__main__":

    start = time.time()

    model = get_model(training=True)

    # prepare train data
    dir_path = "data/train/"
    train = TextImageGenerator(dir_path, img_w, img_h, batch_size, downsample_factor)
    train.build_data()

    # prepare validation data
    dir_path = "data/val/"
    val = TextImageGenerator(dir_path, img_w, img_h, val_batch_size, downsample_factor)
    val.build_data()

    print("\n# of training samples: {}".format(train.n))
    print("\n# of validation samples: {}\n".format(val.n))

    ada = Adadelta(rho=0.9)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=ada)
    checkpoint = ModelCheckpoint(filepath="model--{epoch:02d}.hd5", monitor='loss', mode='min', period=1)

    history = model.fit_generator(generator=train.next_batch(),
                        steps_per_epoch=int(train.n/batch_size),
                        epochs=epochs,
                        callbacks=[checkpoint],
                        validation_data=val.next_batch(),
                        validation_steps=int(val.n/val_batch_size))

    history = history.history

    print("\nTotal train time: {:.2f}s".format(time.time()-start))

    epochs = range(1, len(history['loss'])+1)
    plt.plot(epochs, history['loss'], label="train loss")
    plt.plot(epochs, history['val_loss'], label="val loss")
    plt.legend()
    plt.savefig("model.png")

