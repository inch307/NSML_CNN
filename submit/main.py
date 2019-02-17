# -*- coding: utf_8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import time

import nsml
import numpy as np
import math

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

from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten,
    Dropout,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    Lambda
)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
early_stopper = EarlyStopping(min_delta=0.001, patience=20)

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

        # l2 normalization (when cosine similarity)
        # query_vecs = l2_normalize(query_vecs)
        # reference_vecs = l2_normalize(reference_vecs)

        # Calculate cosine similarity
        # sim_matrix = np.dot(query_vecs, reference_vecs.T)
        # Calculate euclidean distance
        sim_matrix = get_euclidean_dist(query_vecs, reference_vecs)
        indices = np.argsort(sim_matrix, axis=1)
        # flip when cosine simiarity (cos: high rel => high val, distance: high rel => low val)
        # indices = np.flip(indices, axis=1)

        retrieval_results = {}

        for (i, query) in enumerate(queries):
            ranked_list = [references[k] for k in indices[i]]
            ranked_list = ranked_list[:1000]

            retrieval_results[query] = ranked_list
        print('done')

        return list(zip(range(len(retrieval_results)), retrieval_results.items()))

    # DONOTCHANGE: They are reserved for nsml
    nsml.bind(save=save, load=load, infer=infer)


# def l2_normalize(v):
#     norm = np.linalg.norm(v)
#     if norm == 0:
#         return v
#     return v / norm

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

# def l2_normalize(v):
#     norm = np.linalg.norm(v)
#     if norm == 0:
#         return v
#     lst = []
#     for i in v:
#         half = np.split(i, 2)
#         lstAvg = []
#         lstMax = []
#         length = len(i) / 2
#         avgVec = half[0]
#         maxVec = half[1]
#         real_norm_avg = np.linalg.norm(avgVec)
#         real_norm_max = np.linalg.norm(maxVec)
#         if real_norm_avg == 0 and real_norm_max == 0:
#             avgVec /= 1
#             avgVec *= 2
#             maxVec /= 1
#             vec = np.concatenate((avgVec, maxVec), axis=0)
#             lst.append(vec)

#         elif real_norm_max == 0:
#             avgVec /= real_norm_avg
#             avgVec *= 2
#             maxVec /= 1
#             vec = np.concatenate((avgVec, maxVec), axis=0)
#             lst.append(vec)

#         elif real_norm_avg == 0:
#             avgVec /= 1
#             maxVec /= real_norm_max
#             vec = np.concatenate((avgVec, maxVec), axis=0)
#             lst.append(vec)

#         else: 
#             avgVec /= real_norm_avg
#             avgVec *= 2
#             maxVec /= real_norm_max
#             vec = np.concatenate((avgVec, maxVec), axis=0)
#             lst.append(vec)
#     normed_v = np.array(lst)
#     return normed_v

def get_euclidean_dist(q, r):
    result_mat = []
    for vec in q:
        sub = np.subtract(vec, r)
        q_mat = []
        for v in sub:
            sq = np.dot(v, v.T)
            sq = math.sqrt(sq)
            q_mat.append(sq)
        result_mat.append(q_mat)
    result_mat = np.array(result_mat)
    return result_mat

# data preprocess

def get_feature(model, queries, db):
    img_size = (224, 224)
    test_path = DATASET_PATH + '/test/test_data'

    intermediate_layer_model = Model(inputs=model.input, outputs=model.layers[-2].output)
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

if __name__ == '__main__':
    args = argparse.ArgumentParser()

    # hyperparameters
    args.add_argument('--epoch', type=int, default=200)
    args.add_argument('--batch_size', type=int, default=64)
    args.add_argument('--num_classes', type=int, default=1383)

    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0',
                      help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')
    config = args.parse_args()

    # training parameters
    nb_epoch = config.epoch
    batch_size = config.batch_size
    num_classes = config.num_classes
    input_shape = (224, 224, 3)  # input image shape
    image_input = Input(input_shape)
    """ Model """
    model1 = applications.inception_v3.InceptionV3(input_tensor= image_input, include_top= False, weights= None, classes=num_classes)
    last_layer = model1.output
    x1 = GlobalAveragePooling2D()(last_layer)
    # x1 = Lambda(lambda x: x * 2)(x1)
    #x2 = GlobalMaxPooling2D()(last_layer)
    # model2 = applications.densenet.DenseNet121(input_tensor= image_input, include_top= False,pooling='max', classes=num_classes, weights= 'imagenet', input_shape=input_shape)
    #concatenated = concatenate([x1, x2])
    predictions = Dense(num_classes, activation='softmax')(x1)
    model = Model(inputs=image_input, outputs=predictions)
    model.summary()
    bind_model(model)

    if config.pause:
        nsml.paused(scope=locals())

    bTrainmode = False
    if config.mode == 'train':
        bTrainmode = True

        nsml.load(checkpoint='0', session='Avian_Influenza/ir_ph2/115')
        nsml.save('SaeHae-Bok-Many-BaduSae-Yo')
        exit()
