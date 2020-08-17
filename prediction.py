import cv2
import itertools, os
import numpy as np
from utils import letters, img_w, img_h, max_text_len
import argparse
from main import get_model
import pandas as pd
import time

start = time.time()


def decode_label(out_value):
    """
    We skip first two letter predicted because we noticed that model produces garbage values.
    output_str: decoded value predicted by model
    :param out_value:
    :return: output_str
    """
    out_best = list(np.argmax(out_value[0, 2:], axis=1))
    out_best = [k for k, g in itertools.groupby(out_best)]
    output_str = ''
    for i in out_best:
        if i < len(letters):
            output_str += letters[i]
    return output_str


test_dir = "../data/test/"
test_imgs = os.listdir(test_dir)

model = get_model(training=False)
model.load_weights("../model/model_5_itr--20.hd5")


# load test labels
text_data = pd.read_csv("../data/test_label.csv", header=None)


def predict(img):

    total = 0
    acc = 0
    letter_total = 0
    letter_acc = 0
    start = time.time()

    for j, img in enumerate(sorted(test_imgs, key = lambda s : int(s[:-4]))):
    # print("count: {} img: {}".format(j, img))
        img = cv2.imread(os.path.join(test_dir, img), cv2.IMREAD_GRAYSCALE)
        img_pred = img.astype(np.float32)
        img_pred = cv2.resize(img_pred, (img_w, img_h))
        img_pred = (img_pred/255.0) + 2.0 - 1.0
        img_pred = img_pred.T
        img_pred = np.expand_dims(img_pred, axis=-1)
        img_pred = np.expand_dims(img_pred, axis=0)

        net_out_value = model.predict(img_pred)

        pred_texts = decode_label(net_out_value)
        pred_texts = str(pred_texts)
        test_img = text_data.iloc[j:j+1, 0:1].values[0][0]
        test_img = str(test_img)
        crop = max_text_len - len(pred_texts)
        
        #pred_texts = pred_texts[:crop]
        print("Pred Text: {} True Text: {}".format(pred_texts, test_img))

        for i in range(min(len(str(pred_texts)), len(str(test_img)))):
            if pred_texts[i] == test_img[i]:
                letter_acc += 1
            letter_total += max(len(pred_texts), len(test_img))
        
        if pred_texts == test_img:
            # print("here")
            acc += 1
    
        total += 1
        # break
        
    print(total, acc)
    return acc/total, letter_acc/letter_total


if __name__ == "__main__":
    word_acc, letter_acc = predict("16.jpg")
    print("Accuracy: {}".format(word_acc))
    print("letter Accuracy: {}".format(letter_acc))
    print("Total time: {}".format(time.time()-start))



