# -*- coding: utf_8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import cv2
import os
import sys
import random
import os
import argparse
import time

import nsml
import numpy as np

from nsml import DATASET_PATH
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import applications
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.training_utils import multi_gpu_model
from keras.layers.merge import concatenate
from keras.optimizers import Adam

from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten,
    Dropout,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    Concatenate,
    Multiply,
    Lambda
)

def bind_model(model):
    def save(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        model.save_weights(os.path.join(dir_name, 'model'))
        print('model saved!')

    def load(file_path):
        model.load_weights(file_path)
        print('model loaded!')

    def infer(queries, _):
        test_path = DATASET_PATH + '/test/test_data'

        db = [os.path.join(test_path, 'reference', path) for path in os.listdir(os.path.join(test_path, 'reference'))]

        queries = [v.split('/')[-1].split('.')[0] for v in queries]
        db = [v.split('/')[-1].split('.')[0] for v in db]
        queries.sort()
        db.sort()

        queries, query_vecs, references, reference_vecs = get_feature(model, queries, db)

        # l2 normalization
        query_vecs = l2_normalize(query_vecs)
        reference_vecs = l2_normalize(reference_vecs)

        # Calculate cosine similarity
        sim_matrix = np.dot(query_vecs, reference_vecs.T)
        indices = np.argsort(sim_matrix, axis=1)
        indices = np.flip(indices, axis=1)

        retrieval_results = {}

        for (i, query) in enumerate(queries):
            ranked_list = [references[k] for k in indices[i]]
            ranked_list = ranked_list[:1000]

            retrieval_results[query] = ranked_list
        print('done')

        return list(zip(range(len(retrieval_results)), retrieval_results.items()))

    # DONOTCHANGE: They are reserved for nsml
    nsml.bind(save=save, load=load, infer=infer)

def l2_normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    lst = []
    for i in v:
        real_norm = np.linalg.norm(i)
        if real_norm == 0:
            lst.append(i)
        else: lst.append(i / real_norm)
    normed_v = np.array(lst)
    return normed_v

# data preprocess
def get_feature(model, queries, db):
    img_size = (224, 224)
    test_path = DATASET_PATH + '/test/test_data'

    intermediate_layer_model = Model(inputs=model.get_layer(model_1).input, outputs=model.get_layer(model_1).output)
    intermediate_layer_model.trainable = False
    test_datagen = ImageDataGenerator(rescale=1. / 255, dtype='float32')
    query_generator = test_datagen.flow_from_directory(
        directory=test_path,
        target_size=(224, 224),
        classes=['query'],
        color_mode="rgb",
        batch_size=32,
        class_mode=None,
        shuffle=False
    )
    query_vecs = intermediate_layer_model.predict_generator(query_generator, steps=len(query_generator), verbose=1)

    reference_generator = test_datagen.flow_from_directory(
        directory=test_path,
        target_size=(224, 224),
        classes=['reference'],
        color_mode="rgb",
        batch_size=32,
        class_mode=None,
        shuffle=False
    )
    reference_vecs = intermediate_layer_model.predict_generator(reference_generator, steps=len(reference_generator),
                                                                verbose=1)

    return queries, query_vecs, db, reference_vecs
def triplet_gen(anchor_gen, gen):
    while True:
        anchors, y_anc = next(anchor_gen)
        pos = np.empty(anchors.shape)
        neg = np.empty(anchors.shape)
        for sample_idx in range(anchors.shape[0]):
            while pos[sample_idx].any() == None or neg[sample_idx].any() == None:
                img, y = next(gen)
                if y == y_anc[sample_idx]:
                    pos[sample_idx,...] = img
                else:
                    neg[sample_idx,...] = img
        yield [anchors, pos, neg], y_anc

def lossless_triplet_loss(y_true, y_pred, N = 2, beta=2, epsilon=1e-10):
    """
    Implementation of the triplet loss function
    
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor data
            positive -- the encodings for the positive data (similar to anchor)
            negative -- the encodings for the negative data (different from anchor)
    N  --  The number of dimension 
    beta -- The scaling factor, N is recommended
    epsilon -- The Epsilon value to prevent ln(0)
    
    
    Returns:
    loss -- real number, value of the loss
    """
    anchor = tf.convert_to_tensor(y_pred[:,0:N])
    positive = tf.convert_to_tensor(y_pred[:,N:N*2]) 
    negative = tf.convert_to_tensor(y_pred[:,N*2:N*3])
    
    # distance between the anchor and the positive
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,positive)),1)
    # distance between the anchor and the negative
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,negative)),1)
    
    #Non Linear Values  
    
    # -ln(-x/N+1)
    pos_dist = -tf.log(-tf.divide((pos_dist),beta)+1+epsilon)
    neg_dist = -tf.log(-tf.divide((N-neg_dist),beta)+1+epsilon)
    
    # compute loss
    loss = neg_dist + pos_dist
    
    return loss


if __name__ == '__main__':
    args = argparse.ArgumentParser()

    # hyperparameters
    args.add_argument('--epoch', type=int, default=200)
    args.add_argument('--batch_size', type=int, default=16
    )
    args.add_argument('--num_classes', type=int, default=1383)

    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0',
                      help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')
    config = args.parse_args()

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
    nb_epoch = config.epoch
    batch_size = config.batch_size
    num_classes = config.num_classes
    input_shape = (224, 224, 3)  # input image shape
    image_input = Input(shape=input_shape, name='submit_input')
    anc_inp = Input(shape=input_shape, name="anc_inp")
    pos_inp = Input(shape=input_shape, name="pos_inp")
    neg_inp = Input(shape=input_shape, name="neg_inp")
    """ Model """
    print('------------dddd----------------')
    model1 = applications.inception_resnet_v2.InceptionResNetV2(input_tensor= image_input, include_top= False, weights= None, input_shape=input_shape)
     # last layer
    last_layer = model1.output
    last_layer = GlobalAveragePooling2D(name='embedded_value')(last_layer)
    model1 = Model(inputs=image_input, outputs= last_layer)
    bind_model(model1)
    nsml.load(checkpoint='0', session='Avian_Influenza/ir_ph2/151')
    #x1 = Lambda(lambda x: x * 2)(x1)
    #x2 = GlobalMaxPooling2D()(last_layer)
    # model2 = applications.densenet.DenseNet121(input_tensor= image_input, include_top= False,pooling='max', classes=num_classes, weights= 'imagenet', input_shape=input_shape)
    #concatenated = concatenate([x1, x2])
    anc_outp = model1(anc_inp)
    pos_outp = model1(pos_inp)
    neg_outp = model1(neg_inp)
    merged_outp = Concatenate(axis=-1)([anc_outp, pos_outp, neg_outp])
    model = Model(inputs=[anc_inp, pos_inp, neg_inp], outputs=merged_outp)
    adam_fine = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False) #10x smaller than standard
    # model = multi_gpu_model(model, gpus=2)
    model.compile(optimizer='adam', loss=lossless_triplet_loss)
      
    # model.trainable = False
    # model.layers[-1].trainable = True
    # model.layers[1].trainable = False
    
    model.summary()
    bind_model(model)
    nsml.load(checkpoint='7', session='Avian_Influenza/ir_ph2/198')
    nsml.save('toSubmit')
    exit()

    if config.pause:
        nsml.paused(scope=locals())

    bTrainmode = False
    if config.mode == 'train':
        bTrainmode = True

        """ Initiate RMSprop optimizer """

        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            dtype='float32',
            validation_split=0.2)

        # rescale=1. / 255,

        anchor_train_generator = train_datagen.flow_from_directory(
            directory=DATASET_PATH + '/train/train_data',
            target_size=input_shape[:2],
            color_mode="rgb",
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=True,
            seed=42,
            subset='training'
        )
        anchor_validation_generator = train_datagen.flow_from_directory(
            directory=DATASET_PATH + '/train/train_data',
            target_size=input_shape[:2],
            color_mode="rgb",
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=True,
            seed=42,
            subset='validation'
        )
        train_generator = train_datagen.flow_from_directory(
            directory=DATASET_PATH + '/train/train_data',
            target_size=input_shape[:2],
            color_mode="rgb",
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=True,
            seed=42,
            subset='training'
        )
        validation_generator = train_datagen.flow_from_directory(
            directory=DATASET_PATH + '/train/train_data',
            target_size=input_shape[:2],
            color_mode="rgb",
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=True,
            seed=42,
            subset='validation'
        )

        classes_1 = dict((v, k) for k, v in anchor_train_generator.class_indices.items())
        num_classes_1 = len(classes_1)


        """ Training loop """
        STEP_SIZE_TRAIN = anchor_train_generator.n // anchor_train_generator.batch_size
        print('step tr : ', STEP_SIZE_TRAIN)
        print('len ', len(anchor_train_generator))
        STEP_SIZE_VALIDATION = anchor_validation_generator.n // anchor_validation_generator.batch_size
        t0 = time.time()
        for epoch in range(nb_epoch):
            t1 = time.time()
            res = model.fit_generator(generator=triplet_gen(anchor_train_generator, train_generator),
                                      steps_per_epoch=len(anchor_train_generator),
                                      validation_data=triplet_gen(anchor_validation_generator, validation_generator),
                                      validation_steps=len(anchor_validation_generator),
                                      initial_epoch=epoch,
                                      epochs=epoch + 1,
                                      verbose=2,
                                      shuffle=True)
            t2 = time.time()
            print(res.history)
            print('Training time for one epoch : %.1f' % ((t2 - t1)))
            train_loss = res.history['loss'][0]
            nsml.report(summary=True, epoch=epoch, epoch_total=nb_epoch, loss=train_loss)
            nsml.save(epoch)
        print('Total training time : %.1f' % (time.time() - t0))
