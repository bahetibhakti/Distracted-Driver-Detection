

import numpy as np
import os
import pickle
import pandas as pd
import time
import warnings
import cv2
warnings.filterwarnings("ignore")
from numpy.random import permutation
np.random.seed(2016)
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten # GlobalAveragePooling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.utils import np_utils
from keras.models import model_from_json
from sklearn.metrics import log_loss, confusion_matrix
from keras import regularizers
import h5py


from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.advanced_activations import LeakyReLU
from sklearn.metrics import confusion_matrix

from keras.applications.imagenet_utils import _obtain_input_shape
from keras import backend as K
from keras.layers import Input, Convolution2D, \
    GlobalAveragePooling2D, Dense, BatchNormalization, Activation
from keras.models import Model
from keras.engine.topology import get_source_inputs
from depthwise_conv2d import DepthwiseConvolution2D


use_cache = 1

def load_train():
    '''Give path of .csv file of training data below'''
    df = pd.read_csv(r'/home/gpu3/Desktop/mobileVGG/auc.distracted.driver.train.csv')
    x = df.iloc[:,0]
    y = df.iloc[:,1]
    X_train = []
    Y_train = []
    print('Read test images')
    for i in range (0,len(x)):
        fl=x[i]
        img = get_im_cv2(fl)
        X_train.append(img)
        Y_train.append(y[i])
    return X_train, Y_train

def load_valid():
    '''Give path of .csv file of test data below'''
    df = pd.read_csv(r'/home/gpu3/Desktop/mobileVGG/auc.distracted.driver.test.csv')
    x = df.iloc[:,0]
    y = df.iloc[:,1]
    X_valid = []
    Y_valid = []
    print('Read test images')
    for i in range (0,len(x)):
        fl=x[i] 
        img = get_im_cv2(fl)
        X_valid.append(img)
        Y_valid.append(y[i])
    return X_valid, Y_valid


def get_im_cv2(path):
    img = cv2.imread(path)
    resized = cv2.resize(src=img, dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
    return resized

def cache_data(data, path):
    if os.path.isdir(os.path.dirname(path)):
        file = open(path, 'wb')
        pickle.dump(data, file)
        file.close()
    else:
        print('Directory doesnt exists')


def restore_data(path):
    data = dict()
    if os.path.isfile(path):
        file = open(path, 'rb')
        data = pickle.load(file)
    return data
    



def read_and_normalize_train_data():
    cache_path = os.path.join('/home/gpu3/Desktop/mobileVGG','cache', 'train_r_' + str(224) + '_c_' + str(224) + '_t_' + str(3) + '.dat')
    if not os.path.isfile(cache_path) or use_cache == 0:
        train_data, train_target= load_train()
        cache_data((train_data, train_target), cache_path)
    else:
        print('Restore train from cache!')
        (train_data, train_target) = restore_data(cache_path)
    
    print('Convert to numpy...')
    train_data = np.array(train_data, dtype=np.uint8)
    train_target = np.array(train_target, dtype=np.uint8)
    
    print('Reshape...')
    train_data = train_data.transpose((0, 1, 2, 3))

    # Normalise the train data
    print('Convert to float...')
    train_data = train_data.astype('float16')
    mean_pixel = [80.857, 81.106, 82.928]
    
    print('Substract 0...')
    train_data[:, :, :, 0] -= mean_pixel[0]
    
    print('Substract 1...')
    train_data[:, :, :, 1] -= mean_pixel[1]

    print('Substract 2...')
    train_data[:, :, :, 2] -= mean_pixel[2]

    train_target = np_utils.to_categorical(train_target, 10)
    
    # Shuffle experiment START !!!
    perm = permutation(len(train_target))
    train_data = train_data[perm]
    train_target = train_target[perm]
    # Shuffle experiment END !!!
    
    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    return train_data, train_target

def read_and_normalize_test_data():
    start_time = time.time()
    cache_path = os.path.join('/home/gpu3/Desktop/mobileVGG','cache', 'test_r_' + str(224) + '_c_' + str(224) + '_t_' + str(3) + '.dat')

    if not os.path.isfile(cache_path) or use_cache == 0:
        test_data, test_target = load_valid()
        cache_data((test_data, test_target ), cache_path)
    else:
        print('Restore test from cache [{}]!')
        (test_data, test_target) = restore_data(cache_path)

    test_data = np.array(test_data, dtype=np.uint8)
    test_data = test_data.transpose((0, 1, 2, 3))

    # Normalise the test data data

    test_data = test_data.astype('float16')
    mean_pixel = [80.857, 81.106, 82.928]

    test_data[:, :, :, 0] -= mean_pixel[0]

    test_data[:, :, :, 1] -= mean_pixel[1]

    test_data[:, :, :, 2] -= mean_pixel[2]

    test_target = np_utils.to_categorical(test_target, 10)
    print('Test shape:', test_data.shape)
    print(test_data.shape[0], 'test samples')
    print('Read and process test data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return test_data, test_target


def VGG_with_MobileNet(input_tensor=None, input_shape=None, alpha=1, shallow=False, classes=10):


    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=96,
                                      data_format=K.image_data_format(),
                                      include_top=True)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    """ Input and 3x3 conv 64 filters"""
    x = Convolution2D(int(64 * alpha), (3, 3), strides=(1,1), padding='same',W_regularizer=l2(0.001), use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x= LeakyReLU(alpha=0.01)(x)

    """ 3x3 conv 64 filters and maxpooling by 2"""
    x = DepthwiseConvolution2D(int(64 * alpha), (3, 3), strides=(2, 2), padding='same', W_regularizer=l2(0.001), use_bias=False)(x)
    x = BatchNormalization()(x)
    x= LeakyReLU(alpha=0.01)(x)
    
    x = Convolution2D(int(64 * alpha), (1, 1), strides=(1, 1), padding='same',W_regularizer=l2(0.001),  use_bias=False)(x)
    x = BatchNormalization()(x)
    x= LeakyReLU(alpha=0.01)(x)
   
    """ 3x3 conv 128 filters"""   
    x = DepthwiseConvolution2D(int(64 * alpha), (3, 3), strides=(1, 1), padding='same',W_regularizer=l2(0.001), use_bias=False)(x)
    x = BatchNormalization()(x)
    x= LeakyReLU(alpha=0.01)(x)
    
    x = Convolution2D(int(128 * alpha), (1, 1), strides=(1, 1), padding='same', W_regularizer=l2(0.001), use_bias=False)(x)
    x = BatchNormalization()(x)
    x= LeakyReLU(alpha=0.01)(x)
 
    """ 3x3 conv 128 filters and maxpooling by 2"""    
    x = DepthwiseConvolution2D(int(128 * alpha), (3, 3), strides=(2, 2), padding='same', W_regularizer=l2(0.001), use_bias=False)(x)
    x = BatchNormalization()(x)
    x= LeakyReLU(alpha=0.01)(x)
    
    x = Convolution2D(int(128 * alpha), (1, 1), strides=(1, 1), padding='same', W_regularizer=l2(0.001),use_bias=False)(x)
    x = BatchNormalization()(x)
    x= LeakyReLU(alpha=0.01)(x)

    """ 3x3 conv 256 filters"""   
    x = DepthwiseConvolution2D(int(128 * alpha), (3, 3), strides=(1, 1), padding='same',W_regularizer=l2(0.001), use_bias=False)(x)
    x = BatchNormalization()(x)
    x= LeakyReLU(alpha=0.01)(x)
    x = Convolution2D(int(256 * alpha), (1, 1), strides=(1, 1), padding='same',W_regularizer=l2(0.001), use_bias=False)(x)
    x = BatchNormalization()(x)
    x= LeakyReLU(alpha=0.01)(x)

    """ 3x3 conv 256 filters"""   
    x = DepthwiseConvolution2D(int(256 * alpha), (3, 3), strides=(1, 1), padding='same', W_regularizer=l2(0.001),use_bias=False)(x)
    x = BatchNormalization()(x)
    x= LeakyReLU(alpha=0.01)(x)
    x = Convolution2D(int(256 * alpha), (1, 1), strides=(1, 1), padding='same', W_regularizer=l2(0.001),use_bias=False)(x)
    x = BatchNormalization()(x)
    x= LeakyReLU(alpha=0.01)(x)

    """ 3x3 conv 256 filters and maxpooling by 2"""    
    x = DepthwiseConvolution2D(int(256 * alpha), (3, 3), strides=(2, 2), padding='same', W_regularizer=l2(0.001),use_bias=False)(x)
    x = BatchNormalization()(x)
    x= LeakyReLU(alpha=0.01)(x)
    x = Convolution2D(int(256 * alpha), (1, 1), strides=(1, 1), padding='same',W_regularizer=l2(0.001), use_bias=False)(x)
    x = BatchNormalization()(x)
    x= LeakyReLU(alpha=0.01)(x)
    
    x= Dropout(0.05)(x)

    
    """ 3x3 conv 512 filters"""       
    x = DepthwiseConvolution2D(int(256 * alpha), (3, 3), strides=(1, 1), padding='same',W_regularizer=l2(0.001), use_bias=False)(x)
    x = BatchNormalization()(x)
    x= LeakyReLU(alpha=0.01)(x)
    x = Convolution2D(int(512 * alpha), (1, 1), strides=(1, 1), padding='same', W_regularizer=l2(0.001), use_bias=False)(x)
    x = BatchNormalization()(x)
    x= LeakyReLU(alpha=0.01)(x)
    
    """ 3x3 conv 512 filters"""       
    x = DepthwiseConvolution2D(int(512 * alpha), (3, 3), strides=(1, 1), padding='same',W_regularizer=l2(0.001), use_bias=False)(x)
    x = BatchNormalization()(x)
    x= LeakyReLU(alpha=0.01)(x)
    x = Convolution2D(int(512 * alpha), (1, 1), strides=(1, 1), padding='same', W_regularizer=l2(0.001), use_bias=False)(x)
    x = BatchNormalization()(x)
    x= LeakyReLU(alpha=0.01)(x)
    
        
    """ 3x3 conv 512 filters and maxpooling by 2"""       
    x = DepthwiseConvolution2D(int(512 * alpha), (3, 3), strides=(2, 2), padding='same',W_regularizer=l2(0.001), use_bias=False)(x)
    x = BatchNormalization()(x)
    x= LeakyReLU(alpha=0.01)(x)
    x = Convolution2D(int(512 * alpha), (1, 1), strides=(1, 1), padding='same', W_regularizer=l2(0.001), use_bias=False)(x)
    x = BatchNormalization()(x)
    x= LeakyReLU(alpha=0.01)(x)

    x= Dropout(0.1)(x)
    
    """ 3x3 conv 512 filters"""       
    x = DepthwiseConvolution2D(int(512 * alpha), (3, 3), strides=(1, 1), padding='same',W_regularizer=l2(0.001), use_bias=False)(x)
    x = BatchNormalization()(x)
    x= LeakyReLU(alpha=0.01)(x)
    x = Convolution2D(int(512 * alpha), (1, 1), strides=(1, 1), padding='same', W_regularizer=l2(0.001), use_bias=False)(x)
    x = BatchNormalization()(x)
    x= LeakyReLU(alpha=0.01)(x)
    
    """ 3x3 conv 512 filters"""       
    x = DepthwiseConvolution2D(int(512 * alpha), (3, 3), strides=(1, 1), padding='same',W_regularizer=l2(0.001), use_bias=False)(x)
    x = BatchNormalization()(x)
    x= LeakyReLU(alpha=0.01)(x)
    x = Convolution2D(int(512 * alpha), (1, 1), strides=(1, 1), padding='same', W_regularizer=l2(0.001), use_bias=False)(x)
    x = BatchNormalization()(x)
    x= LeakyReLU(alpha=0.01)(x)
    
        
    """ 3x3 conv 512 filters and maxpooling by 2"""       
    x = DepthwiseConvolution2D(int(512 * alpha), (3, 3), strides=(2, 2), padding='same',W_regularizer=l2(0.001), use_bias=False)(x)
    x = BatchNormalization()(x)
    x= LeakyReLU(alpha=0.01)(x)
    x = Convolution2D(int(512 * alpha), (1, 1), strides=(1, 1), padding='same', W_regularizer=l2(0.001), use_bias=False)(x)
    x = BatchNormalization()(x)
    x= LeakyReLU(alpha=0.01)(x)

    x= Dropout(0.2)(x)
    
    """ 7x7 conv 512 filters"""       
    x = DepthwiseConvolution2D(int(512 * alpha), (7, 7), strides=(1, 1), padding='same',W_regularizer=l2(0.001), use_bias=False)(x)
    x = BatchNormalization()(x)
    x= LeakyReLU(alpha=0.01)(x)
    x = Convolution2D(int(512 * alpha), (1, 1), strides=(1, 1), padding='same', W_regularizer=l2(0.001), use_bias=False)(x)
    x = BatchNormalization()(x)
    x= LeakyReLU(alpha=0.01)(x)
    
    x= Dropout(0.3)(x)
    
    """ 7x7 conv 512 filters"""       
    x = Convolution2D(int(512 * alpha), (1, 1), strides=(1, 1), padding='same', W_regularizer=l2(0.001), use_bias=False)(x)
    x = BatchNormalization()(x)
    #x = Activation('relu')(x)
    x= LeakyReLU(alpha=0.01)(x)
    
    x= Dropout(0.4)(x)
    
  
    x = GlobalAveragePooling2D()(x)
    out = Dense(classes, activation='softmax')(x)

    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    model = Model(inputs, out, name='VGG_with_MobileNet')
    model.load_weights('weights.h5');
    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.000001)

    model.compile(adam, loss='categorical_crossentropy',metrics=['accuracy'])
    return model

def run_model():
    batch_size = 32
    nb_epoch = 500

    
    X_train, Y_train = read_and_normalize_train_data()
    X_valid, Y_valid = read_and_normalize_test_data()
    
    
    #Data augmentation
    datagen = ImageDataGenerator(
              width_shift_range=0.2,
              height_shift_range=0.2,
              zoom_range=0.2,
              shear_range=0.2
              )
    
    datagen.fit(X_train)
    model = VGG_with_MobileNet()
    
    weights_path=os.path.join('/home/gpu3/Desktop/mobileVGG','Checkpoint','weights.h5')       
    callbacks = [ModelCheckpoint(weights_path, monitor='val_acc', save_best_only=True, verbose=1)]

    
    hist=model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                    steps_per_epoch=2*len(X_train) / batch_size, nb_epoch=nb_epoch,
           verbose=1, validation_data=(X_valid, Y_valid), callbacks=callbacks)

    pd.DataFrame(hist.history).to_csv("/home/gpu3/Desktop/mobileVGG/cache/try_hist.csv")

    predictions_valid = model.predict(X_valid.astype('float32'), batch_size=batch_size, verbose=1)
    cm1=confusion_matrix(Y_valid.argmax(axis=1), predictions_valid.argmax(axis=1))
    ss=cm1[0,0]+cm1[1,1]+cm1[2,2]+cm1[3,3]+cm1[4,4]+cm1[5,5]+cm1[6,6]+cm1[7,7]+cm1[8,8]+cm1[9,9];
    test_accuracy=np.divide(ss,4331);
    print('Test Accuracy:',test_accuracy)
    
    ppath=os.path.join('/home/gpu3/Desktop/mobileVGG','cache','confusion_mat.npy')
    np.save(ppath, cm1)
    
   

if __name__ == '__main__':
    run_model()
