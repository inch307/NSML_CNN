# -*- coding: utf_8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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


# def l2_normalize(v):
#     norm = np.linalg.norm(v)
#     if norm == 0:
#         return v
#     return v / norm

# def l2_normalize(v, k):
#     norm = np.linalg.norm(v)
#     if norm == 0:
#         return v
#     lst = []
#     for i in v:
#         real_norm = np.linalg.norm(i) / k
#         if real_norm == 0:
#             lst.append(i)
#         else: lst.append(i / real_norm)
#     normed_v = np.array(lst)
#     return normed_v

def l2_normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    lst = []
    for i in v:
        lstAvg = []
        lstMax = []
        length = len(i) / 2
        avgVec = i[0:length]
        maxVec = i[length:]
        real_norm_avg = np.linalg.norm(avgVec)
        real_norm_max = np.linalg.norm(maxVec)
        if real_norm_avg == 0 and real_norm_max == 0:
            avgVec /= 1
            avgVec *= 2
            maxVec /= 1
            vec = np.concatenate((avgVec, maxVec), axis=0)
            lst.append(vec)

        elif real_norm_max == 0:
            avgVec /= real_norm_avg
            avgVec *= 2
            maxVec /= 1
            vec = np.concatenate((avgVec, maxVec), axis=0)
            lst.append(vec)

        elif real_norm_avg == 0:
            avgVec /= 1
            maxVec /= real_norm_max
            vec = np.concatenate((avgVec, maxVec), axis=0)
            lst.append(vec)

        else: 
            avgVec /= real_norm_avg
            avgVec *= 2
            maxVec /= real_norm_max
            vec = np.concatenate((avgVec, maxVec), axis=0)
            lst.append(vec)
    normed_v = np.array(lst)
    return normed_v


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
    model.summary()
    '''

    # training parameters
    nb_epoch = config.epoch
    batch_size = config.batch_size
    num_classes = config.num_classes
    input_shape = (224, 224, 3)  # input image shape
    image_input = Input(input_shape)
    """ Model """
    print('------------dddd----------------')
    model1 = applications.mobilenet_v2.MobileNetV2(input_tensor= image_input, include_top= False, classes=num_classes, weights= 'imagenet', input_shape=input_shape)
    last_layer = model1.output
    x1 = GlobalAveragePooling2D()(last_layer)
    x1 = Lambda(lambda x: x * 2)(x1)
    x2 = GlobalMaxPooling2D()(last_layer)
    # model2 = applications.densenet.DenseNet121(input_tensor= image_input, include_top= False,pooling='max', classes=num_classes, weights= 'imagenet', input_shape=input_shape)
    concatenated = concatenate([x1, x2])
    predictions = Dense(num_classes, activation='softmax')(concatenated)
    model = Model(inputs=image_input, outputs=predictions)
    # model = multi_gpu_model(model, gpus=2)
    model.summary()
    bind_model(model)

    if config.pause:
        nsml.paused(scope=locals())

    bTrainmode = False
    if config.mode == 'train':
        bTrainmode = True

        """ Initiate RMSprop optimizer """
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        print('dataset path', DATASET_PATH)

        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            validation_split=0.05)

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


        """ Training loop """
        STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
        STEP_SIZE_VALIDATION = validation_generator.n // validation_generator.batch_size
        t0 = time.time()
        for epoch in range(nb_epoch):
            t1 = time.time()
            res = model.fit_generator(generator=train_generator,
                                      steps_per_epoch=STEP_SIZE_TRAIN,
                                      validation_data=validation_generator,
                                      validation_steps=STEP_SIZE_VALIDATION,
                                      initial_epoch=epoch,
                                      epochs=epoch + 1,
                                      callbacks=[lr_reducer, early_stopper],
                                      verbose=2,
                                      shuffle=True)
            t2 = time.time()
            print(res.history)
            print('Training time for one epoch : %.1f' % ((t2 - t1)))
            train_loss, train_acc = res.history['loss'][0], res.history['acc'][0]
            nsml.report(summary=True, epoch=epoch, epoch_total=nb_epoch, loss=train_loss, acc=train_acc)
            nsml.save(epoch)
        print('Total training time : %.1f' % (time.time() - t0))
