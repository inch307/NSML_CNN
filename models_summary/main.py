# -*- coding: utf_8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import time
import numpy as np

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import applications
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.training_utils import multi_gpu_model

from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten,
    Dropout,
    GlobalAveragePooling2D
)

if __name__ == '__main__':
    '''
    model = applications.resnet50.ResNet50(input_tensor= image_input, include_top= True, weights= 'imagenet')
    model.summary()
    model = applications.vgg16.VGG16(input_tensor= image_input, include_top= True, weights= 'imagenet')
    model.summary()
    model = applications.vgg19.VGG19(input_tensor= image_input, include_top= True, weights= 'imagenet')
    model.summary()
    model = applications.mobilenet.MobileNet(input_tensor= image_input, include_top= True, weights= 'imagenet')
    model.summary()
    model = applications.densenet.DenseNet121(input_tensor= image_input, include_top= True, weights= 'imagenet')
    model.summary()
    model = applications.densenet.DenseNet169(input_tensor= image_input, include_top= True, weights= 'imagenet')
    model.summary()
    model = applications.densenet.DenseNet201(input_tensor= image_input, include_top= True, weights= 'imagenet')
    model.summary()
    model = applications.mobilenet_v2.MobileNetV2(input_tensor= image_input, include_top= True, weights= 'imagenet')
    applications.inception_resnet_v2.InceptionResNetV2
    applications.inception_v3.InceptionV3
    model.summary()
    '''

    # training parameters
    nb_epoch = 100
    batch_size = 32
    num_classes = 1383
    input_shape = (224, 224, 3)  # input image shape
    image_input = Input(input_shape)
    """ Model """
    model = applications.resnet50.ResNet50(input_tensor= image_input, include_top= False, weights= None, classes=num_classes)
    x = model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=model.input, outputs=predictions)
    model.summary()

    testl = Conv2D(filters=1, kernel_size=(1,1))
    t = type(testl)
    for i in model.layers:
        print(i)
        print(type(i))
        print(type(i) == t)

    # if config.pause:
    #     nsml.paused(scope=locals())

    # bTrainmode = False
    # if config.mode == 'train':
    #     bTrainmode = True

    #     """ Initiate RMSprop optimizer """
    #     model.compile(loss='categorical_crossentropy',
    #                   optimizer='adam',
    #                   metrics=['accuracy'])

    #     print('dataset path', DATASET_PATH)

    #     train_datagen = ImageDataGenerator(
    #         rescale=1. / 255,
    #         validation_split=0.2)

    #     train_generator = train_datagen.flow_from_directory(
    #         directory=DATASET_PATH + '/train/train_data',
    #         target_size=input_shape[:2],
    #         color_mode="rgb",
    #         batch_size=batch_size,
    #         class_mode="categorical",
    #         shuffle=True,
    #         seed=42,
    #         subset='training'
    #     )

    #     validation_generator = train_datagen.flow_from_directory(
    #         directory=DATASET_PATH + '/train/train_data',
    #         target_size=input_shape[:2],
    #         color_mode="rgb",
    #         batch_size=batch_size,
    #         class_mode="categorical",
    #         shuffle=True,
    #         seed=42,
    #         subset='validation'
    #     )


    #     """ Training loop """
    #     STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
    #     STEP_SIZE_VALIDATION = validation_generator.n // validation_generator.batch_size
    #     t0 = time.time()
    #     for epoch in range(nb_epoch):
    #         t1 = time.time()
    #         res = model.fit_generator(generator=train_generator,
    #                                   steps_per_epoch=STEP_SIZE_TRAIN,
    #                                   validation_data=validation_generator,
    #                                   validation_steps=STEP_SIZE_VALIDATION,
    #                                   initial_epoch=epoch,
    #                                   epochs=epoch + 1,
    #                                   callbacks=[lr_reducer, early_stopper],
    #                                   verbose=1,
    #                                   shuffle=True)
    #         t2 = time.time()
    #         print(res.history)
    #         print('Training time for one epoch : %.1f' % ((t2 - t1)))
    #         train_loss, train_acc = res.history['loss'][0], res.history['acc'][0]
    #         nsml.report(summary=True, epoch=epoch, epoch_total=nb_epoch, loss=train_loss, acc=train_acc)
    #         nsml.save(epoch)
    #     print('Total training time : %.1f' % (time.time() - t0))
