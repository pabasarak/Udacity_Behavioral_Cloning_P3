import os
import pickle
import json
import random
import csv

import cv2
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense, ELU
from keras.layers.core import Lambda
from keras.callbacks import Callback
from keras.utils import np_utils

_index_in_epoch = 0

nb_epoch = 20

batch_size = 64

img_height, img_width = 64, 64

global bad
bad = 0


def shuffle(x, y):
    perm = np.arange(len(x))
    np.random.shuffle(perm)
    x = x[perm]
    y = y[perm]

    return (x, y)

def train_test_split(X, Y):
    count = int(len(X)*.7)

    X_train = X[:count]
    Y_train = Y[:count]

    X_val = X[count:]
    Y_val = Y[count:]

    return (X_train, Y_train, X_val, Y_val)

def load_training_and_validation():
    rows, labels = [], []
    with open('data/driving_log3.csv', 'r') as _f:
        reader = csv.reader(_f, delimiter=',')
        next(reader, None)
        for row in reader:
            rows.append(row[0].strip())
            labels.append(float(row[3]))
            # left camera
            rows.append(row[1].strip())
            labels.append(float(row[3])+.15)
            # right camera 
            rows.append(row[2].strip())
            labels.append(float(row[3])-0.15)

    assert len(rows) == len(labels), 'unbalanced data'
    print("logdata1 length",len(rows))
    # shuffle the data
    X, Y = shuffle(np.array(rows), np.array(labels))

    # split into training and validation
    return train_test_split(X, Y)

def resize_image(img):
    #return img
    return cv2.resize(img,(64, 64))  




def next_batch(data, labels, batch_size):
    """Return the next `batch_size` examples from this data set."""
    global _index_in_epoch
    start = _index_in_epoch
    _index_in_epoch += batch_size
    _num_examples = len(data)

    if _index_in_epoch > _num_examples:
        # Shuffle the data
        data, labels = shuffle(data, labels)
        # Start next epoch
        start = 0
        _index_in_epoch = batch_size
        assert batch_size <= _num_examples

    end = _index_in_epoch
    return data[start:end], labels[start:end]


def transform_generator(x, y, batch_size=32):
    global bad
    while True:
        images, labels = list(), list()

        _images, _labels = next_batch(x, y, batch_size)

        #current = os.path.dirname(os.path.realpath(__file__))
        #print('./data2/'+_images[1])
        for i in range(len(_images)):
            #img = cv2.imread('{}/data/{}'.format(current, _images[i]), 1)
            img = cv2.imread('./data/'+_images[i])
            angle = _labels[i]

            if img is None:
                bad += 1
                continue

            if angle != 0:
                resized = resize_image(img)
                
                images.append(resized)
                
                labels.append(angle)

            
            
            

        X = np.array(images, dtype=np.float64).reshape((-1, img_height, img_width, 3))

        Y = np.array(labels, dtype=np.float64)

        yield (X, Y)
        

def gen_nvidia_model():
    ch, row, col = 3, 64, 64  # camera format

    model = Sequential()

    model.add(Lambda(lambda x: x/127.5 - 1.,
            input_shape=(row, col, ch),
            output_shape=(row, col, ch)))

    # trim the hood of the car
    model.add(Lambda(lambda x: x[:,30:-8,:,:]))

    model.add(Convolution2D(32, 3, 3, subsample=(1, 1)))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1)))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    
    model.add(Convolution2D(128, 3, 3, subsample=(1, 1)))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    adam = Adam(lr=0.0001)

    model.compile(loss='mean_squared_error', optimizer=adam)

    return model

def main():
    import tensorflow as tf
    tf.python.control_flow_ops = tf

    X_train, Y_train, X_val, Y_val = load_training_and_validation() 

    assert len(X_train) == len(Y_train), 'unbalanced training data'
    assert len(X_val) == len(Y_val), 'unbalanced validation data'

    print(len(X_train), "training images and ", len(X_val), "validation images")

    model = gen_nvidia_model()
    print("Model Gen Done")

    filepath = "./model.h5"
    checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=True)

    print("filepath ok")

    if not os.path.exists("./outputs/sim"):
            os.makedirs("./outputs/sim")

    print("Model fit starting")

    model.fit_generator(
        transform_generator(X_train, Y_train),
        samples_per_epoch=(len(X_train)*2),
        nb_epoch=nb_epoch,
        validation_data=transform_generator(X_val, Y_val),
        nb_val_samples=len(X_val),
        callbacks=[checkpoint])

    print("Saving model weights and configuration file. and we have {} bad files".format(bad))
    model.save_weights("./model.h5", True)
    with open('./model.json', 'w') as outfile:
        json.dump(model.to_json(), outfile)
        

        
if __name__ == '__main__':
    print("Starting Script")
    main()