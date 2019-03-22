import os
import sys
import random
import warnings
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

import tensorflow as tf

from unet import UNet
from metrics import dice_coef,recall,precision,f1


IMG_WIDTH = 1024
IMG_HEIGHT = 1024
IMG_CHANNELS = 3
TRAIN_PATH = './augmented/'

seed = 42
random.seed = seed
np.random.seed = seed

im_list=glob.glob(TRAIN_PATH+'images/*')
mask_list=glob.glob(TRAIN_PATH+'masks/*')

X_train=[]
Y_train=[]

width=1024

for n, id_ in tqdm(enumerate(im_list), total=len(im_list)):
    im=cv2.imread(im_list[n])
    im=cv2.resize(im,(width,width),interpolation = cv2.INTER_CUBIC)
    X_train.append(im)
    
for n, id_ in tqdm(enumerate(mask_list), total=len(mask_list)):
    mask=cv2.imread(mask_list[n],0)
    mask=cv2.resize(mask,(width,width),interpolation = cv2.INTER_CUBIC)
    Y_train.append(mask)

X_train=np.array(X_train)
Y_train=np.array(Y_train)

Y_train=Y_train.reshape(Y_train.shape+(1,))

model=UNet((width,width))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mse',dice_coef])
model.summary()

earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint('unet.{epoch:02d}-{val_loss:.2f}.h5',monitor='val_dice_coef', verbose=1, save_best_only=True)
results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=100, 
                    callbacks=[earlystopper, checkpointer])
