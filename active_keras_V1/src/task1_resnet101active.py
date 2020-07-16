# -*- coding: utf-8 -*-
import pdb
import uuid
import tensorflow as tf
# from keras.models import load_model
import keras
from keras.models import Sequential
from keras.optimizers import SGD,Adam
import argparse
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, Reshape, Activation,add
from keras.layers import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam, SGD, RMSprop
from keras.backend import tensorflow_backend as K
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import log_loss
from keras.callbacks import TensorBoard, LearningRateScheduler, ModelCheckpoint,EarlyStopping,ReduceLROnPlateau,CSVLogger, Callback,LambdaCallback
from custom_layers.scale_layer import Scale
from keras.utils.layer_utils import convert_all_kernels_in_model
import numpy as np
import datetime
import os
import tqdm
from keras import metrics
import sys
import collections
import matplotlib.image as img
# import cv2
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.regularizers import l2
from keras.applications.imagenet_utils import preprocess_input
from keras.utils import np_utils

from PIL import Image
import matplotlib.pyplot as plt
import random

#for diversity
import pandas as pd
from LocalitySensitiveHashing import *
from pandas import read_csv
from sklearn.decomposition import PCA
from PIL import Image
import csv
# 

sys.setrecursionlimit(3000)
currtime = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
run_id = str(uuid.uuid4())

def np_onehot(arr, n_sample,n_class):
    n_class = arr.max() + 1
    n_sample = arr.shape[0]
    array_one_hot = np.zeros((n_sample, n_class))
    array_one_hot[:, arr] = 1
    return array_one_hot

def get_data_label(data_dir,batch_size):
    data_lists = []
    label_lists = []
    data_label_lists,_ = get_data_label_list(data_dir)
    for epoch in range(int(len(data_label_lists)//batch_size)):
        for i in range(batch_size):
            data_label = data_label_lists[epoch*batch_size+i]
            img_RGB=Image.open(data_label['file_path'])
            img_RGB_resize= img_RGB.resize((224,224),Image.ANTIALIAS)
            img = np.asarray(img_RGB_resize)#change to numpy
            data_lists.append(img)
            label_lists.append(data_label['label'])
    datas = np.array(data_lists)
    labels = np.array(label_lists)
    return datas,labels

def get_data_label_val(csv_path, data_dir,batch_size):
    data_lists = []
    label_lists = []
    data_label_lists,_ = get_data_label_list_fromcsv(csv_path,data_dir)
    for epoch in range(int(len(data_label_lists)//batch_size)):
        for i in range(batch_size):
            data_label = data_label_lists[epoch*batch_size+i]
            img_RGB=Image.open(data_label['file_path'])
            img_RGB_resize= img_RGB.resize((224,224),Image.ANTIALIAS)
            img = np.asarray(img_RGB_resize)#change to numpy
            data_lists.append(img)
            label_lists.append(data_label['label'])
    datas = np.array(data_lists)
    labels = np.array(label_lists)
    return datas,labels

def get_data_label_predict(model_new,data_dir,batch_size):
    import copy
    data_lists = []
    label_lists = []
    pred_val_lists = []
    pred_pro_lists = []
    data_label_lists,_ = get_data_label_list(data_dir)
    # random.shuffle(data_label_lists)
    # pdb.set_trace()
    for epoch in range(int(len(data_label_lists)//batch_size)):
        for i in range(batch_size):
            data_label = data_label_lists[epoch*batch_size+i]
            img_RGB=Image.open(data_label['file_path'])
            img_RGB_resize= img_RGB.resize((224,224),Image.ANTIALIAS)
            img = np.asarray(img_RGB_resize,'f')#change to numpy
            # img_preprocess = copy.deepcopy(img)
            # preprocess_input(img_preprocess,mode='tf')
            img_preprocess = preprocess_input(img)
            pred_pro = np.array(model_new.predict(np.expand_dims(img_preprocess, axis=0), batch_size=None, verbose=0, steps=1))
            # pdb.set_trace()
            pred_val=np.argmax(pred_pro,axis=1)
            pred_val_lists.append(list(pred_val))
            data_lists.append(img)
            label_lists.append(data_label['label'])
    datas = np.array(data_lists)
    labels = np.array(label_lists)
    pred_val = np.array(pred_val_lists)
    acc,correct_numbers = precision(pred_val,labels)
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(labels, pred_val).ravel()
    return acc,correct_numbers

def one_hot(labels):
        if labels.shape[-1] == 1:
            labels = np.reshape(labels, (-1))
        max_label = np.max(labels) + 1
        return np.eye(max_label)[labels]

def get_data_label_predict_general(model_new,data_dir,num_classes):
    from sklearn.metrics import confusion_matrix,roc_auc_score,average_precision_score,mean_squared_error
    import copy
    data_lists = []
    label_lists = []
    pred_val_lists = []
    pos_pred_prob_lists = []
    pos_pro_lists = []
    data_label_lists,_ = get_data_label_list(data_dir)
    # random.shuffle(data_label_lists)
    # pdb.set_trace()
    for epoch in range(int(len(data_label_lists))):
        data_label = data_label_lists[epoch]
        img_RGB=Image.open(data_label['file_path'])
        img_RGB_resize= img_RGB.resize((224,224),Image.ANTIALIAS)
        img = np.asarray(img_RGB_resize,'f')#change to numpy
        img_preprocess = preprocess_input(img)
        pred_pro = np.array(model_new.predict(np.expand_dims(img_preprocess, axis=0), batch_size=None, verbose=0, steps=1))
        pos_pred_prob_lists.append(pred_pro[:,1])
        pred_val=np.argmax(pred_pro,axis=1)
        pred_val_lists.append(list(pred_val))
        data_lists.append(img)
        label_lists.append(data_label['label'])
    datas = np.array(data_lists)
    labels = np.array(label_lists)
    pred_vals = np.array(pred_val_lists)
    pos_pred_probs = np.array(pos_pred_prob_lists)
    acc,correct_numbers = precision(pred_vals,labels)
    tn, fp, fn, tp = confusion_matrix(labels, pred_vals).ravel()
    total_tp = np.nonzero(labels)[0].shape[0]
    sensitivity = tp/(tp+fn)
    specifity = tn/(tn+fp)
    auc = roc_auc_score(labels, pos_pred_probs)
    m_ap = average_precision_score(labels, pos_pred_probs)
    result = 1 * np.logical_xor(labels, pred_vals.squeeze())
    print ('Standard Error:', np.std(result))
    print ("acc:",acc)
    print ("auc:",auc)
    print ("m_ap:",m_ap)
    print ("sensitivity:",sensitivity)
    print ("specifity:",specifity)
    #pdb.set_trace()
    return acc,auc,m_ap,sensitivity,specifity

def get_generator_predict_general(model_new,data_dir,batch_size):
    from sklearn.metrics import confusion_matrix,roc_auc_score,average_precision_score
    import copy
    steps_done = 0
    label_lists = []
    steps = batch_size
    cycle = 32
    pred_pros_many = np.zeros((cycle,batch_size,2),dtype=np.float)
    # test_datagen = ImageDataGenerator( 
    #                                   preprocessing_function=preprocess_input)
    # test_datagen = ImageDataGenerator( 
    #                                   preprocessing_function=preprocess_input,
    #                                   horizontal_flip=True,
    #                                   # vertical_flip=True,
    #                                   # shear_range=0.2,
    #                                   # rotation_range=90,
    #                                   zoom_range=[.8, 1.2],
    #                                   fill_mode='constant',
    #                                   cval=0.
    #                                   )
    test_datagen = ImageDataGenerator( 
                                      preprocessing_function=preprocess_input,
                                      rotation_range=90,
                                      # horizontal_flip=True,  # randomly flip images
                                      # vertical_flip=True,  # randomly flip images
                                      # shear_range=0.2,
                                      # zoom_range=[.8, 1.2],
                                      # fill_mode='constant',
                                      # cval=0.
                                      )
    
    # test_datagen = ImageDataGenerator(
    #                 featurewise_center=False,  # set input mean to 0 over the dataset
    #                 samplewise_center=False,  # set each sample mean to 0
    #                 featurewise_std_normalization=False,  # divide inputs by std of the dataset
    #                 samplewise_std_normalization=False,  # divide each input by its std
    #                 zca_whitening=False,  # apply ZCA whitening
    #                 rotation_range=90,
    #                 # rotation_range=40,
    #                 shear_range=0.2,
    #                 preprocessing_function=preprocess_input,
    #                 horizontal_flip=True,  # randomly flip images
    #                 vertical_flip=True,  # randomly flip images
    #                 zoom_range=[.8, 1.2],
    #                 # zoom_range=[.8, .8],
    #                 # zoom_range=[.5, .5],
    #                 # zoom_range=[.2, .2],
    #                 # width_shift_range=0.25,
    #                 # height_shift_range=0.25,
    #                 # fill_mode='nearest',
    #                 fill_mode='constant',
    #                 cval=0.)
        
    test_generator = test_datagen.flow_from_directory(data_dir, target_size=(224, 224),
                                                           batch_size=batch_size,shuffle = False,
                                                           class_mode='categorical')
    # one test_generator batch  batch = batch_size
    # j = 0 
    # for i in range (cycle):
    #     while steps_done < (batch_size):
    #         generator = next(test_generator)
    #         if isinstance(generator, tuple):
    #             # Compatibility with the generators
    #             # used for training.
    #             if len(generator) == 2:
    #                 x, y = generator
    #                 label_lists.append(y[:,1])
    #                 pred_pros_many[i,:,:] = model_new.predict(x,steps = 1)
    #                 j+=1
    #         steps_done+=1
    #     filenames = test_generator.filenames
    #     nb_samples = len(filenames)

    # pred_pros = np.squeeze(np.mean(pred_pros_many,axis=0))
    # pos_pred_probs = pred_pros[:,1]
    # pred_vals=np.argmax(pred_pros,axis=1)
    # labels = np.array(label_lists)
    
    while steps_done < (steps // batch_size):
        generator = next(test_generator)
        if isinstance(generator, tuple):
            # Compatibility with the generators
            # used for training.
            if len(generator) == 2:
                x, y = generator
        steps_done+=1
    filenames = test_generator.filenames
    nb_samples = len(filenames)
    # pdb.set_trace()
    for i in range(cycle):
        pred_pros_many[i,:,:] = model_new.predict_generator(test_generator,steps = 1)
    pred_pros = np.squeeze(np.mean(pred_pros_many,axis=0))
    pos_pred_probs = pred_pros[:,1]
    pred_vals=np.argmax(pred_pros,axis=1)
    labels = y[:,1]
    

    # pred_vals = np.array(pred_val_lists)
    pdb.set_trace()
    acc,correct_numbers = precision(pred_vals,labels)
    tn, fp, fn, tp = confusion_matrix(labels, pred_vals).ravel()
    total_tp = np.nonzero(labels)[0].shape[0]
    sensitivity = tp/(tp+fn)
    specifity = tn/(tn+fp)
    # pdb.set_trace()
    auc = roc_auc_score(labels, pos_pred_probs)
    m_ap = average_precision_score(labels, pos_pred_probs)
    print ("acc:",acc)
    print ("auc:",auc)
    print ("m_ap:",m_ap)
    print ("sensitivity:",sensitivity)
    print ("specifity:",specifity)
    # pdb.set_trace()


def get_data_label_predictseveral_general(model_news,data_dir,num_classes):
    from sklearn.metrics import confusion_matrix,roc_auc_score,average_precision_score
    import copy
    data_lists = []
    label_lists = []
    pred_val_lists = []
    pos_pred_prob_lists = []
    pos_pro_lists = []
    data_label_lists,_ = get_data_label_list(data_dir)
    # random.shuffle(data_label_lists)
    # pdb.set_trace()
    for epoch in range(int(len(data_label_lists))):
        data_label = data_label_lists[epoch]
        img_RGB=Image.open(data_label['file_path'])
        img_RGB_resize= img_RGB.resize((224,224),Image.ANTIALIAS)
        img = np.asarray(img_RGB_resize,'f')#change to numpy
        img_preprocess = preprocess_input(img)
        pred_pro1 = np.array(model_news[0].predict(np.expand_dims(img_preprocess, axis=0), batch_size=None, verbose=0, steps=1))
        pred_pro2 = np.array(model_news[1].predict(np.expand_dims(img_preprocess, axis=0), batch_size=None, verbose=0, steps=1))
        pred_pro3 = np.array(model_news[2].predict(np.expand_dims(img_preprocess, axis=0), batch_size=None, verbose=0, steps=1))
        pred_pro = np.mean(np.concatenate((pred_pro1,pred_pro2,pred_pro3),axis=0),axis=0)
        pos_pred_prob_lists.append(pred_pro[1])
        pred_val=np.argmax(pred_pro,axis=0)
        pred_val_lists.append(pred_val.tolist())
        data_lists.append(img)
        label_lists.append(data_label['label'])
    datas = np.array(data_lists)
    labels = np.array(label_lists)
    pred_vals = np.array(pred_val_lists)
    pos_pred_probs = np.array(pos_pred_prob_lists)
    acc,correct_numbers = precision(pred_vals,labels)
    tn, fp, fn, tp = confusion_matrix(labels, pred_vals).ravel()
    total_tp = np.nonzero(labels)[0].shape[0]
    sensitivity = tp/(tp+fn)
    specifity = tn/(tn+fp)
    pdb.set_trace()
    auc = roc_auc_score(labels, pos_pred_probs)
    m_ap = average_precision_score(labels, pos_pred_probs)
    print ("acc:",acc)
    print ("auc:",auc)
    print ("m_ap:",m_ap)
    print ("sensitivity:",sensitivity)
    print ("specifity:",specifity)
    pdb.set_trace()
    return acc,correct_numbers

def get_aug_data_label_predict_general(model_new,data_dir,aug_times,num_classes):
    from sklearn.metrics import confusion_matrix,roc_auc_score,average_precision_score
    from utils import np_onehot,Aug_test
    from keras.utils import np_utils
    import copy
    is_erase = False
    is_flip = True
    is_tpswrap = False
    data_lists = []
    label_lists = []
    pred_val_lists = []
    pos_pred_prob_lists = []
    data_label_lists,_ = get_data_label_list(data_dir)
    aug_datas_labels = []
    #pred all list, each sample predict aug_times times
    for i in range(len(data_label_lists)):
        for j in range(aug_times):
            aug_datas_labels.extend(data_label_lists[i])
    # random.shuffle(data_label_lists)
    # pdb.set_trace()
    for epoch in range(int(len(data_label_lists))):
        #each sample predict aug_times times
        # pred_vals = np.zeros((aug_times, 1))
        # pred_pros = np.zeros((aug_times, 1))
        img_augs = np.zeros((aug_times, 224,224,3))
        data_label = data_label_lists[epoch]
        # y_onehot = np_utils.to_categorical(data_label['label'], num_classes)
        # y_onehot.shape=(num_classes,1)
        for j in range(aug_times):
            img_RGB=Image.open(data_label['file_path'])
            img_RGB_resize= img_RGB.resize((224,224),Image.ANTIALIAS)
            img = np.asarray(img_RGB_resize,'f')#change to numpy
            img_preprocess = preprocess_input(img)
            Aug = Aug_test(model_new)
            img_augs[j,:,:,:] = Aug.aug(img_preprocess,is_erase, is_tpswrap, is_flip)
        # batch predict after each aug
        pred_pros = model_new.predict(img_augs, batch_size=64)  #64*2
        pred_pro = np.mean(pred_pros,axis=0) #all
        pos_pred_pro = np.mean(pred_pros[:, 1],axis=0) #positive 
        pred_val=np.argmax(pred_pro,axis=0) #1
        pos_pred_prob_lists.append(pos_pred_pro.tolist())
        pred_val_lists.append(pred_val.tolist())
        label_lists.append(data_label['label'])
    pred_vals = np.array(pred_val_lists)
    pos_pred_probs = np.array(pos_pred_prob_lists)
    labels = np.array(label_lists)
    pdb.set_trace()
    acc,correct_numbers = precision(pred_vals,labels)
    tn, fp, fn, tp = confusion_matrix(labels, pred_vals).ravel()
    total_tp = np.nonzero(labels)[0].shape[0]
    sensitivity = tp/(tp+fn)
    specifity = tn/(tn+fp)
    auc = roc_auc_score(labels, pos_pred_probs)
    m_ap = average_precision_score(labels, pos_pred_probs)
    print ("acc:",acc)
    print ("auc:",auc)
    print ("m_ap:",m_ap)
    print ("sensitivity:",sensitivity)
    print ("specifity:",specifity)
    pdb.set_trace()
    return acc,correct_numbers


def list_predict(model_new,data_dir,unselected,batch_size):
    #select data from data_dir within val_list
    # pred_val_lists = []
    pred_val_dicts = {}
    data_label_lists,_ = get_data_label_list(data_dir)
    # random.shuffle(data_label_lists)
    for i in range(len(unselected)):
        data_label = data_label_lists[unselected[i]] #unselected[i]corresponds to the index in data_label_lists
        img_RGB=Image.open(data_label['file_path'])
        img_RGB_resize= img_RGB.resize((224,224),Image.ANTIALIAS)
        img = np.asarray(img_RGB_resize,'f')#change to numpy
        img_preprocess = preprocess_input(img)
        pred_pro = np.array(model_new.predict(np.expand_dims(img_preprocess, axis=0), batch_size=None, steps=1))
        pred_val_dicts[str(unselected[i])]=np.max(pred_pro)
        # interest_layer = [layer for layer in model_new.layers if
        #        'fc8' in layer.name]

        # pred_pro = np.array(model_new.predict_proba(np.expand_dims(img_preprocess, axis=0), batch_size=None, steps=1))
        # inp = model_new.layers[0].input
        # out = interest_layer[0].output
        # functors = K.function([inp]+ [K.learning_phase()], [out])
        # if i == 0:
        #     features = np.asarray(functors([np.expand_dims(img_preprocess, axis=0), 0.])[0])
        #     # pdb.set_trace()
        # else:
        #     features = np.append(features,np.asarray(functors([np.expand_dims(img_preprocess, axis=0), 0.])[0]),axis=0)

        # xx = model_new.output.op.inputs[0]
        # xx = interest_layer[0].output.op.inputs[0]
        # func = K.function([model_new.layers[0].input] + [K.learning_phase()], [xx])
        # outxx = func([np.expand_dims(img_preprocess, axis=0), 0.])
        # print (outxx)


        # model.output.op.inputs[0]
        #'avg_pool', 'flatten_1', 'fc8'
        # print (output)
        # pred_val=np.argmax(pred_val,axis=1)
        
    return pred_val_dicts

def list_generator_predict(model_new,data_dir,unselected,batch_size):
    #select data from data_dir within val_list
    pred_val_dicts = {}
    data_label_lists,_ = get_data_label_list(data_dir)
    imgs = np.zeros((len(unselected),224,224,3),dtype=float)
    labels = np.zeros((len(unselected),1))
    # random.shuffle(data_label_lists)
    for i in range(len(unselected)):
        data_label = data_label_lists[unselected[i]] #unselected[i]corresponds to the index in data_label_lists
        img_RGB=Image.open(data_label['file_path'])
        img_RGB_resize= img_RGB.resize((224,224),Image.ANTIALIAS)
        img = np.asarray(img_RGB_resize,'f')#change to numpy
        imgs[i,:,:,:] = img
        labels[i] = data_label['label']
        # img_preprocess = preprocess_input(img)
        # pred_pro = np.array(model_new.predict(np.expand_dims(img_preprocess, axis=0), batch_size=None, steps=1))
        # pred_val_dicts[str(unselected[i])]=np.max(pred_pro)  
    # data augumentation   
    vote_datagen = ImageDataGenerator(featurewise_center=False,  # set input mean to 0 over the dataset
                                     samplewise_center=False,  # set each sample mean to 0
                                     featurewise_std_normalization=False,  # divide inputs by std of the dataset
                                     samplewise_std_normalization=False,  # divide each input by its std
                                     zca_whitening=False,  # apply ZCA whitening
                                     preprocessing_function=preprocess_input,
                                     horizontal_flip=True,  # randomly flip images
                                     zoom_range=[.8, 1.2],
                                     fill_mode='constant',
                                     cval=0.
                                     )
    labels_onehot = np_utils.to_categorical(labels, 2)
    # generate all the data and predict in a batch 
    vote_datagen.fit(imgs)
    '''
    labels = np.ones((3,1),dtype=int)
    img1 = np.ones((224,224,3),dtype=float)
    img2 = np.ones((224,224,3),dtype=float) * 2
    img3 = np.ones((224,224,3),dtype=float) * 3
    imgs = np.concatenate((np.expand_dims(img1,axis=0),np.expand_dims(img2,axis=0),np.expand_dims(img3,axis=0)),axis=0)
    labels_onehot = np_utils.to_categorical(labels, 2)
    vote_datagen.fit(imgs)
    '''
    for i, (X_batch, y_batch) in enumerate(vote_datagen.flow(imgs, 
                                              labels_onehot, 
                                              shuffle=False, 
                                              batch_size=1)):
        print ("round ",i)
        pred_pro = np.array(model_new.predict(X_batch))
        pred_val_dicts[str(unselected[i])]=np.max(pred_pro)
        if i==(len(unselected)-1):
            return pred_val_dicts

def list_predict_diversity(data_dir,unselected):
    #select data from data_dir within val_list
    # pred_val_lists = []
    pred_val_dicts = {}
    pca_dicts=[]
    data_label_lists,_ = get_data_label_list(data_dir)
    n_components=1
    # random.shuffle(data_label_lists)
    for i in range(len(unselected)):
        data_label = data_label_lists[unselected[i]] #unselected[i]corresponds to the index in data_label_lists
        img_Gray=Image.open(data_label['file_path']).convert('L')
        img_Gray_resize= img_Gray.resize((224,224),Image.ANTIALIAS)
        img = np.asarray(img_Gray_resize,'f')#change to numpy
        pca = PCA(n_components=n_components)
        fit = pca.fit(img)
        flat_components_=fit.components_.flatten()
        pca_result_as_list=flat_components_.tolist()
        pca_result_as_list.insert(0,'name_%s'%str(i))
        pca_dicts.append(pca_result_as_list)
    my_df=pd.DataFrame(pca_dicts)
    my_df.to_csv('aaa.csv',index=False,header=False)
    datafile = "aaa.csv"
    dim=224*n_components #multiply pca n_components
    lsh = LocalitySensitiveHashing(
                   datafile = datafile,
                   dim = dim,
                   r = 30,
                   b = 10,
                   expected_num_of_clusters = 20,
          )
    lsh.get_data_from_csv()
    lsh.initialize_hash_store()
    lsh.hash_all_data()
    similarity_groups = lsh.lsh_basic_for_neighborhood_clusters()
    coalesced_similarity_groups = lsh.merge_similarity_groups_with_coalescence( similarity_groups )
    merged_similarity_groups = lsh.merge_similarity_groups_with_l2norm_sample_based( coalesced_similarity_groups )
    lsh.write_clusters_to_file( similarity_groups, "similarity.txt" )
    lsh.write_clusters_to_file( coalesced_similarity_groups, "coalesced.txt" )
    lsh.write_clusters_to_file( merged_similarity_groups, "merged.txt" )
    return merged_similarity_groups

def list_predict_for_fun(model_new,data_dir,unselected,batch_size):
    #select data from data_dir within val_list
    # pred_val_lists = []
    pred_val_dicts = {}
    data_label_lists,_ = get_data_label_list(data_dir)
    # random.shuffle(data_label_lists)
    for i in range(len(unselected)):
        data_label = data_label_lists[unselected[i]] #unselected[i]corresponds to the index in data_label_lists
        img_RGB=Image.open(data_label['file_path'])
        img_RGB_resize= img_RGB.resize((224,224),Image.ANTIALIAS)
        img = np.asarray(img_RGB_resize,'f')#change to numpy
        img_preprocess = preprocess_input(img)
        pred_pro = np.array(model_new.predict(np.expand_dims(img_preprocess, axis=0), batch_size=None, steps=1))
        pred_class=np.argmax(pred_pro,axis=1)
        pred_val_dicts[str(unselected[i])]=[np.max(pred_pro),pred_class]

        # interest_layer = [layer for layer in model_new.layers if
        #        'fc8' in layer.name]

        # pred_pro = np.array(model_new.predict_proba(np.expand_dims(img_preprocess, axis=0), batch_size=None, steps=1))
        # inp = model_new.layers[0].input
        # out = interest_layer[0].output
        # functors = K.function([inp]+ [K.learning_phase()], [out])
        # if i == 0:
        #     features = np.asarray(functors([np.expand_dims(img_preprocess, axis=0), 0.])[0])
        #     # pdb.set_trace()
        # else:
        #     features = np.append(features,np.asarray(functors([np.expand_dims(img_preprocess, axis=0), 0.])[0]),axis=0)

        # xx = model_new.output.op.inputs[0]
        # xx = interest_layer[0].output.op.inputs[0]
        # func = K.function([model_new.layers[0].input] + [K.learning_phase()], [xx])
        # outxx = func([np.expand_dims(img_preprocess, axis=0), 0.])
        # print (outxx)


        # model.output.op.inputs[0]
        #'avg_pool', 'flatten_1', 'fc8'
        # print (output)
        # pred_val=np.argmax(pred_val,axis=1)
        
    return pred_val_dicts

#label of benign is 0, maligant is 1
def get_data_label_list(path):
    # pdb.set_trace()
    _, foldernames, files = next(os.walk(path))
    data_label_dicts = []
    foldernames.sort()
    assert (foldernames==['benign', 'malignant'] or foldernames==['nonsk', 'sk'])
    #match the label with the foldernames
    for i in range(len(foldernames)):
        print ('label %s:' % str(i), foldernames[i])
    per_class_number = [0 for i in range(len(foldernames))]
    for folder_index, file in enumerate(foldernames):
        if file == foldernames[0]:
            label = 0
        if file == foldernames[1]:
            label = 1
        class_path = os.path.join(path,file)
        c_root,c_dirs,c_files = next(os.walk(class_path))
        c_files.sort()
        for index, class_file in enumerate(c_files):
            data_label_dicts.append({'label':label,'file_path':os.path.join(class_path,class_file)})
            per_class_number[folder_index] += 1
    return data_label_dicts,per_class_number

def recursive_data_list(path):
    import os
    import fnmatch
    i = 0
    img_names = []
    for root, dir, files in os.walk(path):
        print (root)
        print ("")
        for items in fnmatch.filter(files, "*"):
                print (i,items)
                img_names.append(items.split('.')[0])
                i+=1
        print ("")
    return img_names

def get_data_label_list_fromcsv(csv_label_path,img_path):
    import csv
    data_label_dicts = []
    per_class_number = [0 for i in range(2)]
    with open(csv_label_path) as csvfile:
        reader = csv.DictReader(csvfile)
        # rows = list(reader)
        # totalrows = len(rows)
        for row in reader:
            ## keys() == odict_keys(['image_id', 'melanoma', 'seborrheic_keratosis'])
            label = int(float(row['melanoma']))
            if label == 1 :
                per_class_number[1] += 1
            if label == 0 :
                per_class_number[0] += 1
            file_name = row['image_id']
            data_label_dicts.append({'label':label,'file_path':os.path.join(img_path,file_name)})
    return data_label_dicts,per_class_number

def transfer_fromcsvTofolder(csv_label_path,img_path,save_dir):
    import csv
    import scipy.misc
    data_label_dicts = []
    per_class_number = [0 for i in range(2)]
    with open(csv_label_path) as csvfile:
        reader = csv.DictReader(csvfile)
        # rows = list(reader)
        # totalrows = len(rows)
        for row in reader:
            ## keys() == odict_keys(['image_id', 'melanoma', 'seborrheic_keratosis'])
            label = int(float(row['seborrheic_keratosis']))
            file_name = row['image_id']
            file_path = os.path.join(img_path,file_name)
            img_RGB=Image.open(file_path+'.jpg')
            img_RGB_resize= img_RGB.resize((224,224),Image.ANTIALIAS)
            if label == 0 :
                save_path = os.path.join(save_dir,'nonsk',file_name+'.jpg')
                scipy.misc.imsave(save_path,img_RGB_resize)
            if label == 1 :
                save_path = os.path.join(save_dir,'sk',file_name+ '.jpg')
                scipy.misc.imsave(save_path,img_RGB_resize)

def precision(test_y,predicted_class):
    # pdb.set_trace()
    # np.savetxt("2_test_y.txt",test_y)
    # np.savetxt("2_label_y.txt",predicted_class)
    correct = (np.squeeze(test_y) == predicted_class)
    acc = correct.mean()*100
    correct_numbers = correct.sum()
    return acc,correct_numbers

def preprocess3(out):
    out=img_to_array(out)

    out = out[:, :, [2, 1, 0]]
    random_crop_size = (224, 224)

    w, h = out.shape[0], out.shape[1]
    rangew = (w - random_crop_size[0]) // 2
    rangeh = (h - random_crop_size[1]) // 2
    offsetw = 0 if rangew == 0 else np.random.randint(rangew)
    offseth = 0 if rangeh == 0 else np.random.randint(rangeh)
    return array_to_img(out[offsetw:offsetw + random_crop_size[0], offseth:offseth + random_crop_size[1], :],scale=False)

def preprocess(out):
    mean = np.array([103.939, 116.779, 123.68],dtype=float)
    # img->arr
    out=img_to_array(out)
    # 'RGB'->'BGR'
    out = out[:, :, [2, 1, 0]]
    # minus junzhi
    out[:, :, 0] -= mean[0]
    out[:, :, 1] -= mean[1]
    out[:, :, 2] -= mean[2]
    return out

def preprocess1(out):
    # img->arr
    out=img_to_array(out)
    # minus junzhi
    out /= 127.5
    out -= 1.
    return out

def build_file_name(loc,currtime):
    model_path = (os.path.dirname(os.path.realpath(__file__)) 
        +loc + '_' + currtime)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    return model_path
def resnet101_model_mixup(img_rows, img_cols, color_type=1, num_classes=None):
    """
    Resnet 101 Model for Keras
    Model Schema and layer naming follow that of the original Caffe implementation
    https://github.com/KaimingHe/deep-residual-networks
    ImageNet Pretrained Weights
    Theano: https://drive.google.com/file/d/0Byy2AcGyEVxfdUV1MHJhelpnSG8/view?usp=sharing
    TensorFlow: https://drive.google.com/file/d/0Byy2AcGyEVxfTmRRVmpGWDczaXM/view?usp=sharing
    Parameters:
      img_rows, img_cols - resolution of inputs
      channel - 1 for grayscale, 3 for color
      num_classes - number of class labels for our classification task
    """
    eps = 1.1e-5
    mn = 0.99
    weight_decay = 1e-4
    # Handle Dimension Ordering for different backends
    global bn_axis
    if K.image_dim_ordering() == 'tf':
      bn_axis = 3
      img_input = Input(shape=(img_rows, img_cols, color_type), name='data')
    else:
      bn_axis = 1
      img_input = Input(shape=(color_type, img_rows, img_cols), name='data')

    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=False,kernel_regularizer = l2(weight_decay))(x)
    x = BatchNormalization(epsilon=eps, momentum=mn,axis=bn_axis, name='bn_conv1')(x)
    x = Scale(axis=bn_axis, name='scale_conv1')(x)
    x = Activation('relu', name='conv1_relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    for i in range(1,4):
      x = identity_block(x, 3, [128, 128, 512], stage=3, block='b'+str(i))

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    for i in range(1,23):
      x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b'+str(i))

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x_fc = AveragePooling2D((7, 7), name='avg_pool')(x)
    x_fc = Flatten()(x_fc)
    x_fc = Dense(1000, activation='relu', name='fc1000',kernel_regularizer = l2(weight_decay))(x_fc)
    x_fc = Dropout(0.2)(x_fc)
    x_fc = Dense(num_classes, activation='softmax', name='fc8')(x_fc)
    # model = Model(img_input, x_fc)
    # model.load_weights(weights_path, by_name=True)

    # # Truncate and replace softmax layer for transfer learning
    # # Cannot use model.layers.pop() since model is not of Sequential() type
    # # The method below works since pre-trained weights are stored in layers but not in the model
    # x_newfc = AveragePooling2D((7, 7), name='avg_pool')(x)
    # x_newfc = Flatten()(x_newfc)
    # x_newfc = Dense(num_classes, activation='relu', name='fc8')(x_newfc)

    model = Model(img_input, x_fc)
    #for layer in model.layers:
        #if hasattr(layer, 'kernel_regularizer'):

            #layer.kernel_regularizer = l2(1)
    return model
def identity_block(input_tensor, kernel_size, filters, stage, block):
    '''The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    '''
    eps = 1.1e-5
    mn=0.99
    weight_decay=1e-4
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a', use_bias=False,kernel_regularizer = l2(weight_decay))(input_tensor)
    x = BatchNormalization(epsilon=eps, momentum=mn,axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
    x = Activation('relu', name=conv_name_base + '2a_relu')(x)

    x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
    x = Conv2D(nb_filter2,( kernel_size, kernel_size),
                      name=conv_name_base + '2b', use_bias=False,kernel_regularizer = l2(weight_decay))(x)
    x = BatchNormalization(epsilon=eps,momentum=mn, axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = Activation('relu', name=conv_name_base + '2b_relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=False,kernel_regularizer = l2(weight_decay))(x)
    x = BatchNormalization(epsilon=eps,momentum=mn, axis=bn_axis, name=bn_name_base + '2c')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)

    x = add([x, input_tensor], name='res' + str(stage) + block)
    x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    '''conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    '''
    eps = 1.1e-5
    mn = 0.99
    weight_decay = 1e-4
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), strides=strides,
                      name=conv_name_base + '2a', use_bias=False,kernel_regularizer = l2(weight_decay))(input_tensor)
    x = BatchNormalization(epsilon=eps,momentum=mn, axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
    x = Activation('relu', name=conv_name_base + '2a_relu')(x)

    x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
    x = Conv2D(nb_filter2, (kernel_size, kernel_size),
                      name=conv_name_base + '2b', use_bias=False,kernel_regularizer = l2(weight_decay))(x)
    x = BatchNormalization(epsilon=eps, momentum=mn,axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = Activation('relu', name=conv_name_base + '2b_relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=False,kernel_regularizer = l2(weight_decay))(x)
    x = BatchNormalization(epsilon=eps,momentum=mn, axis=bn_axis, name=bn_name_base + '2c')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)

    shortcut = Conv2D(nb_filter3, (1, 1), strides=strides,
                             name=conv_name_base + '1', use_bias=False,kernel_regularizer = l2(weight_decay))(input_tensor)
    shortcut = BatchNormalization(epsilon=eps, momentum=mn,axis=bn_axis, name=bn_name_base + '1')(shortcut)
    shortcut = Scale(axis=bn_axis, name=scale_name_base + '1')(shortcut)

    x = add([x, shortcut], name='res' + str(stage) + block)
    x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    return x

def resnet101_model(img_rows, img_cols, color_type=1, num_classes=None):
    """
    Resnet 101 Model for Keras
    Model Schema and layer naming follow that of the original Caffe implementation
    https://github.com/KaimingHe/deep-residual-networks
    ImageNet Pretrained Weights
    Theano: https://drive.google.com/file/d/0Byy2AcGyEVxfdUV1MHJhelpnSG8/view?usp=sharing
    TensorFlow: https://drive.google.com/file/d/0Byy2AcGyEVxfTmRRVmpGWDczaXM/view?usp=sharing
    Parameters:
      img_rows, img_cols - resolution of inputs
      channel - 1 for grayscale, 3 for color
      num_classes - number of class labels for our classification task
    """
    eps = 1.1e-5
    mn = 0.99
    weight_decay = 1e-4
    # Handle Dimension Ordering for different backends
    global bn_axis
    if K.image_dim_ordering() == 'tf':
      bn_axis = 3
      img_input = Input(shape=(img_rows, img_cols, color_type), name='data')
    else:
      bn_axis = 1
      img_input = Input(shape=(color_type, img_rows, img_cols), name='data')

    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=False,kernel_regularizer = l2(weight_decay))(x)
    x = BatchNormalization(epsilon=eps, momentum=mn,axis=bn_axis, name='bn_conv1')(x)
    x = Scale(axis=bn_axis, name='scale_conv1')(x)
    x = Activation('relu', name='conv1_relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    for i in range(1,4):
      x = identity_block(x, 3, [128, 128, 512], stage=3, block='b'+str(i))

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    for i in range(1,23):
      x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b'+str(i))

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x_fc = AveragePooling2D((7, 7), name='avg_pool')(x)
    x_fc = Flatten()(x_fc)
    x_fc = Dense(1000, activation='softmax', name='fc1000',kernel_regularizer = l2(weight_decay))(x_fc)

    model = Model(img_input, x_fc)
    
    if K.image_dim_ordering() == 'th':
      # Use pre-trained weights for Theano backend
      weights_path = '../model/resnet101_weights_th.h5'
    else:
      # Use pre-trained weights for Tensorflow backend
      weights_path = '../model/resnet101_weights_tf.h5'

    model.load_weights(weights_path, by_name=True)

    # Truncate and replace softmax layer for transfer learning
    # Cannot use model.layers.pop() since model is not of Sequential() type
    # The method below works since pre-trained weights are stored in layers but not in the model
    # pdb.set_trace()
    # for k in tf.all_variables():
    #     print (k)
    x_newfc = AveragePooling2D((7, 7), name='avg_pool')(x)
    x_newfc = Flatten()(x_newfc)
    x_newfc = Dense(num_classes, activation='softmax', name='fc8')(x_newfc)
    model = Model(img_input, x_newfc)
    #for layer in model.layers:
        #if hasattr(layer, 'kernel_regularizer'):

            #layer.kernel_regularizer = l2(1)
    
    # Learning rate is changed to 0.001
    # sgd = SGD(lr=3e-4, decay=0, momentum=0, nesterov=False)
    # model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model
def resnet101_pretrain_model(img_rows, img_cols, color_type=1, num_classes=None):
    """
    Resnet 101 Model for Keras
    Model Schema and layer naming follow that of the original Caffe implementation
    https://github.com/KaimingHe/deep-residual-networks
    ImageNet Pretrained Weights
    Theano: https://drive.google.com/file/d/0Byy2AcGyEVxfdUV1MHJhelpnSG8/view?usp=sharing
    TensorFlow: https://drive.google.com/file/d/0Byy2AcGyEVxfTmRRVmpGWDczaXM/view?usp=sharing
    Parameters:
      img_rows, img_cols - resolution of inputs
      channel - 1 for grayscale, 3 for color
      num_classes - number of class labels for our classification task
    """
    eps = 1.1e-5
    mn = 0.99
    weight_decay = 1e-4
    # Handle Dimension Ordering for different backends
    global bn_axis
    if K.image_dim_ordering() == 'tf':
      bn_axis = 3
      img_input = Input(shape=(img_rows, img_cols, color_type), name='data')
    else:
      bn_axis = 1
      img_input = Input(shape=(color_type, img_rows, img_cols), name='data')

    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=False,kernel_regularizer = l2(weight_decay))(x)
    x = BatchNormalization(epsilon=eps, momentum=mn,axis=bn_axis, name='bn_conv1')(x)
    x = Scale(axis=bn_axis, name='scale_conv1')(x)
    x = Activation('relu', name='conv1_relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    for i in range(1,4):
      x = identity_block(x, 3, [128, 128, 512], stage=3, block='b'+str(i))

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    for i in range(1,23):
      x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b'+str(i))

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x_fc = AveragePooling2D((7, 7), name='avg_pool')(x)
    x_fc = Flatten()(x_fc)
    x_fc = Dense(1000, activation='softmax', name='fc1000',kernel_regularizer = l2(weight_decay))(x_fc)

    model = Model(img_input, x_fc)
    
    if K.image_dim_ordering() == 'th':
      # Use pre-trained weights for Theano backend
      weights_path = '../model/resnet101_weights_th.h5'
    else:
      # Use pre-trained weights for Tensorflow backend
      weights_path = '../model/resnet101_weights_tf.h5'

    model.load_weights(weights_path, by_name=True)

    return model

def resnet101_model_new(img_rows, img_cols, color_type=1, num_classes=None):
    """
    Resnet 101 Model for Keras
    Model Schema and layer naming follow that of the original Caffe implementation
    https://github.com/KaimingHe/deep-residual-networks
    ImageNet Pretrained Weights
    Theano: https://drive.google.com/file/d/0Byy2AcGyEVxfdUV1MHJhelpnSG8/view?usp=sharing
    TensorFlow: https://drive.google.com/file/d/0Byy2AcGyEVxfTmRRVmpGWDczaXM/view?usp=sharing
    Parameters:
      img_rows, img_cols - resolution of inputs
      channel - 1 for grayscale, 3 for color
      num_classes - number of class labels for our classification task
    """
    eps = 1.1e-5
    mn = 0.99
    weight_decay = 1e-4
    # Handle Dimension Ordering for different backends
    global bn_axis
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
        img_input = Input(shape=(img_rows, img_cols, color_type), name='data')
    else:
        bn_axis = 1
        img_input = Input(shape=(color_type, img_rows, img_cols), name='data')
    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=False,kernel_regularizer = l2(weight_decay))(x)
    x = BatchNormalization(epsilon=eps, momentum=mn,axis=bn_axis, name='bn_conv1')(x)
    x = Scale(axis=bn_axis, name='scale_conv1')(x)
    x = Activation('relu', name='conv1_relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    for i in range(1,4):
      x = identity_block(x, 3, [128, 128, 512], stage=3, block='b'+str(i))

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    for i in range(1,23):
      x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b'+str(i))

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    # Truncate and replace softmax layer for transfer learning
    # Cannot use model.layers.pop() since model is not of Sequential() type
    # The method below works since pre-trained weights are stored in layers but not in the model
    # pdb.set_trace()
    # for k in tf.all_variables():
    #     print (k)
    x_newfc = AveragePooling2D((7, 7), name='avg_pool')(x)
    x_newfc = Flatten()(x_newfc)
    x_newfc = Dense(num_classes, activation='softmax', name='fc8')(x_newfc)
    model = Model(img_input, x_newfc)
    return model

def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=10):
    '''
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    '''

    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch / step_size))

    return LearningRateScheduler(schedule)

def val(model_path):
    data_path = '/Users/liyi/Downloads/drive-download-20190905T074850Z-001/2017_ISBI_suply/224data/task1'
    print ('val data_path:',data_path)
    model_new = resnet101_model_new(224, 224, 3, 2)
    #model_new = resnet101_model_mixup(224, 224, 3, 2)
    model_new.load_weights(model_path)
    
    select_data_dir = data_path + '/test'
    num_classes = 2
    batch_size = 600
    
    pred_val_lists = []
    label_lists = []
    # acc,correct_numbers = get_data_label_predict(model_new,select_data_dir,batch_size)
    acc,auc,m_ap,sensitivity,specifity = get_data_label_predict_general(model_new,select_data_dir,num_classes)
    print(acc, auc, m_ap, sensitivity, specifity)
    # get_generator_predict_general(model_new,select_data_dir,batch_size)
    # pdb.set_trace()

# def val(model_path):
#     data_path = '../../../../2017_ISBI_suply/224data/task1'
#     print ('val data_path:',data_path)
#     # data_path = '../../../../2017_ISBI_suply/group_data_random'
#     config = tf.ConfigProto()
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#     config.gpu_options.per_process_gpu_memory_fraction = 0.95
#     session = tf.Session(config=config)
#     K.set_session(session)

#     model_new = resnet101_model_new(224, 224, 3, 2)
#     # model_new2 = resnet101_model_new(224, 224, 3, 2)
#     # model_new3 = resnet101_model_new(224, 224, 3, 2)
#     # model_new1.load_weights(model_paths[0])
#     # model_new2.load_weights(model_paths[1])
#     # model_new3.load_weights(model_paths[2])
#     # model_news = [model_new1,model_new2,model_new3]
#     # model_new = resnet101_model_mixup(224, 224, 3, 2)
#     select_data_dir = data_path + '/val'

#     # select_data_dir = data_path + '/train1400'
#     num_classes = 2
    
#     pred_val_lists = []
#     label_lists = []
#     # acc,correct_numbers = get_data_label_predict(model_new,select_data_dir,batch_size)
#     acc,auc,m_ap,sensitivity,specifity = get_data_label_predict_general(model_new,select_data_dir,num_classes)
#     # acc,correct_numbers = get_data_label_predictseveral_general(model_news,select_data_dir,num_classes)
#     pdb.set_trace()


def aug_val(model_path):
    config = tf.ConfigProto()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    session = tf.Session(config=config)
    K.set_session(session)

    model_new = resnet101_model_new(224, 224, 3, 2)
    model_new.load_weights(model_path)
    # model_new = resnet101_model_mixup(224, 224, 3, 2)

    data_path = '../../../../2017_ISBI_suply/224data/task1'
    select_data_dir = data_path + '/val'
    aug_datas_labels = []
    aug_times = 64
    num_classes = 2
    acc,correct_numbers = get_aug_data_label_predict_general(model_new,select_data_dir,aug_times,num_classes)

    #ly
    #data_label_lists,per_class_number = get_data_label_list(test_dir)

    pdb.set_trace()


def val_for_selection(model_path, unselected):
    data_path = '/Users/liyi/Downloads/drive-download-20190905T074850Z-001/2017_ISBI_suply/224data/task1'
    print ('val_for_selection data_path:',data_path)
    # config = tf.ConfigProto(device_count={'gpu':0})
    # config.gpu_options.per_process_gpu_memory_fraction = 0.95
    # session = tf.Session(config=config)
    # K.set_session(session)

    model_new = resnet101_model_new(224, 224, 3, 2)
    model_new.load_weights(model_path)
    select_data_dir = data_path + '/train'
    batch_size = 1
    
    pred_val_lists = []
    label_lists = []
    # pred_val_dicts = list_predict(model_new,select_data_dir, unselected, batch_size)
    pred_val_dicts = list_generator_predict(model_new,select_data_dir, unselected, batch_size)
    return pred_val_dicts

def val_for_selection_for_fun(model_path, unselected):
    data_path = '/Users/liyi/Downloads/drive-download-20190905T074850Z-001/2017_ISBI_suply/224data/task1'
    print ('val_for_selection_for_fun data_path:',data_path)
    # config = tf.ConfigProto(device_count={'gpu':0})
    # config.gpu_options.per_process_gpu_memory_fraction = 0.95
    # session = tf.Session(config=config)
    # K.set_session(session)

    model_new = resnet101_model_new(224, 224, 3, 2)
    model_new.load_weights(model_path)
    select_data_dir = data_path + '/train'
    batch_size = 1
    
    pred_val_lists = []
    label_lists = []
    pred_val_dicts = list_predict_for_fun(model_new,select_data_dir, unselected, batch_size)
    return pred_val_dicts

def val_for_selection_d(unselected):
    data_path = '/Users/liyi/Downloads/drive-download-20190905T074850Z-001/2017_ISBI_suply/224data/task1'
    print ('val_for_selection_d data_path:',data_path)
    select_data_dir = data_path + '/train'

    bucket_list = list_predict_diversity(select_data_dir, unselected)
    return bucket_list

def get_vgg_features_useless(feature_arr,VGG_PATH,val_data_dir,batch_size):
    from keras.applications.vgg19 import VGG19
    img_rows, img_cols, img_channel = 224, 224, 3
    vgg_base_model = VGG19(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, img_channel))
    weights = [layer.output for layer in vgg_base_model.layers if
               'block5_conv4' in layer.name]
    #print all layers in vgg_base_model
    # print ([layer.name for layer in vgg_base_model.layers])
    # pdb.set_trace()
    interest_layer = [layer for layer in vgg_base_model.layers if
               'block5_conv4' in layer.name]
    inp = interest_layer[0].input
    out = interest_layer[0].output

    # inp = vgg_base_model.layers[0].input
    functors = K.function([inp]+ [K.learning_phase()], [out])  # evaluation functions
    features = [np.zeros((batch_size,14,14,1024),dtype=float)]
    data_label_lists,_ = get_data_label_list(val_data_dir)
    for epoch in range(int(len(data_label_lists)//batch_size)):
        for i in range(batch_size):
            imgs = np.zeros((batch_size, 14,14,1024),dtype=int)
            data_label = data_label_lists[epoch*batch_size+i]
            img_RGB=Image.open(data_label['file_path'])
            img_RGB_resize= img_RGB.resize((224,224),Image.ANTIALIAS)
            img = np.asarray(img_RGB_resize,'f')#change to numpy
            # outrgb = img[:, :, [2, 1, 0]]
            img_preprocess = preprocess_input(img)
            imgs[i] = img_preprocess
        if epoch == 0:
            features = np.asarray(functors([imgs, 0.])[0])
            # pdb.set_trace()
        else:
            features = np.append(features,np.asarray(functors[0]([imgs, 0.])[0]),axis=0)
        print ("epoch:",epoch,"len(data_label_lists):",int(len(data_label_lists)))
    layer_outs = np.asarray(functors([feature_arr, 0.]))[0]

    return layer_outs

def get_vgg_features(VGG_PATH,val_data_dir,batch_size):
    from keras.applications.vgg19 import VGG19
    img_rows, img_cols, img_channel = 224, 224, 3
    vgg_base_model = VGG19(weights='imagenet', include_top=True, 
                            input_shape=(img_rows, img_cols, img_channel), classes=2)
    weights = [layer.output for layer in vgg_base_model.layers if
               'block5_conv4' in layer.name]
    #save the summary
    # with open('vgg_report.txt','w') as fh:
    #     # Pass the file handle in as a lambda function to make it callable
    #     vgg_base_model.summary(print_fn=lambda x: fh.write(x + '\n'))
    # pdb.set_trace()

    #print all layers in vgg_base_model
    # print ([layer.name for layer in vgg_base_model.layers])
    # print(vgg_base_model.summary())
    # pdb.set_trace()

    interest_layer = [layer for layer in vgg_base_model.layers if
               'block5_conv4' in layer.name]

    inp = vgg_base_model.layers[0].input
    out = interest_layer[0].output

    # inp = vgg_base_model.layers[0].input
    functors = K.function([inp]+ [K.learning_phase()], [out])  # evaluation functions
    features = [np.zeros((batch_size,14,14,1024),dtype=float)]
    data_label_lists,_ = get_data_label_list(val_data_dir)
    for epoch in range(int(len(data_label_lists)//batch_size)):
        for i in range(batch_size):
            imgs = np.zeros((batch_size, img_rows, img_cols, img_channel),dtype=int)
            data_label = data_label_lists[epoch*batch_size+i]
            img_RGB=Image.open(data_label['file_path'])
            img_RGB_resize= img_RGB.resize((224,224),Image.ANTIALIAS)
            img = np.asarray(img_RGB_resize,'f')#change to numpy
            # outrgb = img[:, :, [2, 1, 0]]
            img_preprocess = preprocess_input(img)
            imgs[i] = img_preprocess
        if epoch == 0:
            features = np.asarray(functors([imgs, 0.])[0])
            # pdb.set_trace()
        else:
            features = np.append(features,np.asarray(functors([imgs, 0.])[0]),axis=0)
        print ("epoch:",epoch,"len(features):",int(len(features)))

    return features

def get_model_feature(model_path, unselected):
    data_path = '../../../../2017_ISBI_suply/224data/task1'
    print ('get_model_feature data_path:',data_path)
    config = tf.ConfigProto(device_count={'gpu':0})
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    session = tf.Session(config=config)
    K.set_session(session)

    model_new = resnet101_model_new(224, 224, 3, 2)
    model_new.load_weights(model_path)
    select_data_dir = data_path + '/train'
    batch_size = 1
    
    pred_val_lists = []
    label_lists = []

    #ly
    #interest_layer = [layer for layer in vgg_base_model.layers if
    #           'fc8' in layer.name]

    pred_val_dicts = list_predict(model_new,select_data_dir, unselected, batch_size)

    return pred_val_dicts

def display_activation(activations, col_size, row_size): 
    activation = activations
    activation_index=0
    fig1, ax1 = plt.subplots(row_size, col_size, figsize=(row_size*25,col_size*15))
    fig2, ax2 = plt.subplots(row_size, col_size, figsize=(row_size*25,col_size*15))
    pdb.set_trace()
    i = 100
    for row in range(0,row_size): 
      for col in range(0,col_size):
        ax1[row][col].imshow(activation[0, :, :, activation_index+i], cmap='gray')
        ax2[row][col].imshow(activation[1, :, :, activation_index+i], cmap='gray')
        activation_index += 1
    plt.show()
    pdb.set_trace()
# def myprint(s):
#     with open('modelsummary.txt','w+') as f:
#         print(s, file=f)
def feature_comparison(model_path,vgg_path):
    # from resnet101-256-lr_second_fulldata_valauc import resnet101_model
    data_path = '../../../../2017_ISBI_suply/224data/task1'
    print ('visualize_feature data_path:',data_path)
    val_data_dir = data_path + '/test'
    batch_size = 30

    config = tf.ConfigProto(device_count={'gpu':0})
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    session = tf.Session(config=config)
    K.set_session(session)

    model_pretrain = resnet101_pretrain_model(224, 224, 3, 1000)
    model = resnet101_model(224, 224, 3, 2)
    model_new = resnet101_model_new(224, 224, 3, 2)
    model.load_weights(model_path)

    with open('pretrained_res_report.txt','w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model_pretrain.summary(print_fn=lambda x: fh.write(x + '\n'))
    pdb.set_trace()

    # for layer in model_new.layers:
    #     print (layer.name)
    # pdb.set_trace()

    # for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
    #     print(var)

    
    inp = model_new.layers[0].input   # input placeholder
    assert len(inp.shape) == 4 
    input_shape_list = inp.shape.as_list()
    input_shape = (batch_size,input_shape_list[1],input_shape_list[2],input_shape_list[3])
    # reslayers = ['res4b22_relu','res4b22_branch2c']
    outputs = [layer.output for layer in model_new.layers if
               'res4b22_relu' in layer.name]
    # outputs = [layer.output for layer in model_new.layers]          # all layer outputs
    functors = [K.function([inp]+ [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

    #calculate the res_features
    features = [np.zeros((30,14,14,1024),dtype=float)]
    data_label_lists,_ = get_data_label_list(val_data_dir)
    for epoch in range(int(len(data_label_lists)//batch_size)):
        for i in range(batch_size):
            imgs = np.zeros(input_shape,dtype=int)
            data_label = data_label_lists[epoch*batch_size+i]
            img_RGB=Image.open(data_label['file_path'])
            img_RGB_resize= img_RGB.resize((224,224),Image.ANTIALIAS)
            img = np.asarray(img_RGB_resize,'f')#change to numpy
            # outrgb = img[:, :, [2, 1, 0]]
            img_preprocess = preprocess_input(img)
            imgs[i] = img_preprocess
        if epoch == 0:
            features = np.asarray(functors[0]([imgs, 0.])[0])
            # pdb.set_trace()
        else:
            features = np.append(features,np.asarray(functors[0]([imgs, 0.])[0]),axis=0)
        print ("epoch:",epoch,"len(features):",int(len(features)))
    print ("res4b22_relu shape:", features.shape)
    pdb.set_trace()
    pretrained_res_features = features

    #calculate the pretrained_res_features
    inp = model_pretrain.layers[0].input   # input placeholder
    assert len(inp.shape) == 4 
    input_shape_list = inp.shape.as_list()
    input_shape = (batch_size,input_shape_list[1],input_shape_list[2],input_shape_list[3])
    # reslayers = ['res4b22_relu','res4b22_branch2c']
    outputs = [layer.output for layer in model_pretrain.layers if
               'res4b22_relu' in layer.name]
    # outputs = [layer.output for layer in model_pretrain.layers]          # all layer outputs
    functors = [K.function([inp]+ [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

    #calculate the res_features
    features = [np.zeros((30,14,14,1024),dtype=float)]
    data_label_lists,_ = get_data_label_list(val_data_dir)
    for epoch in range(int(len(data_label_lists)//batch_size)):
        for i in range(batch_size):
            imgs = np.zeros(input_shape,dtype=int)
            data_label = data_label_lists[epoch*batch_size+i]
            img_RGB=Image.open(data_label['file_path'])
            img_RGB_resize= img_RGB.resize((224,224),Image.ANTIALIAS)
            img = np.asarray(img_RGB_resize,'f')#change to numpy
            # outrgb = img[:, :, [2, 1, 0]]
            img_preprocess = preprocess_input(img)
            imgs[i] = img_preprocess
        if epoch == 0:
            features = np.asarray(functors[0]([imgs, 0.])[0])
            # pdb.set_trace()
        else:
            features = np.append(features,np.asarray(functors[0]([imgs, 0.])[0]),axis=0)
        print ("epoch:",epoch,"len(features):",int(len(features)))
    print ("pretrained_res4b22_relu shape:", features.shape)
    pdb.set_trace()
    res_features = features

    #calculate the vgg_features 
    vgg_features = get_vgg_features(VGG_PATH=vgg_path,val_data_dir=val_data_dir,batch_size=batch_size) #vgg5_4(600,14,14,512)
    print ("block5_conv4 shape:", vgg_features.shape)
    pdb.set_trace()

    #calculate the dice coefficence of vgg_feature and res_feature in same scale (batch_size,14,14,1024) & (batch_size,14,14,512)
    # feature = np.mean(features, axis=-1)
    # vgg_feature = np.mean(vgg_features, axis=-1)
    # aaa = features[:,:,:,0:512] - vgg_features
    featureminus = pretrained_res_features - res_features
    pdb.set_trace()
    display_activation(featureminus, 10, 10)
    pdb.set_trace()
    # aaa = []
    # for i in range(2):
    #     aaa.append(features[:,:,:,0:512] - vgg_features)
    #     pdb.set_trace()

def visualize_feature(model_path,selected,data_label_lists):
    label_lists = []
    batch_size = 1

    # for fear of reload the same graph, we first reset graph into null
    keras.backend.clear_session()
    model_new = resnet101_model_new(224, 224, 3, 2)
    model_new.load_weights(model_path)

    inp = model_new.layers[0].input   # input placeholder
    assert len(inp.shape) == 4 
    input_shape_list = inp.shape.as_list()
    input_shape = (batch_size,input_shape_list[1],input_shape_list[2],input_shape_list[3])
    # reslayers = ['res4b22_relu','res4b22_branch2c','res5c','res5c_relu','avg_pool',
    # 'flatten_1','fc8']
    visual_layer = 'flatten_1' #batch_size*2048
    outputs = [layer.output for layer in model_new.layers if
               visual_layer in layer.name]
    # outputs = [layer.output for layer in model_new.layers]          # all layer outputs
    functors = [K.function([inp]+ [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

    #calculate the res_features
    features = [np.zeros((batch_size,2048),dtype=float)]
    for i in range(len(selected)):
        data_label = data_label_lists[selected[i]]
        imgs = np.zeros(input_shape,dtype=int)
        label_lists.append(data_label['label'])
        img_RGB=Image.open(data_label['file_path'])
        img_RGB_resize= img_RGB.resize((224,224),Image.ANTIALIAS)
        img = np.asarray(img_RGB_resize,'f')#change to numpy
        img_preprocess = preprocess_input(img)
        if i == 0:
            features = np.asarray(functors[0]([np.expand_dims(img_preprocess, axis=0), 0.])[0])
        else:
            features = np.append(features,np.asarray(functors[0]([np.expand_dims(img_preprocess, axis=0), 0.])[0]),axis=0)
        print ("interation:",i,"len(features):",int(len(features)))
    labels = np.array(label_lists)
    print (visual_layer, features.shape)
    return features,labels

def visualize_feature_for_group(model_path,group_list_dict,img_size,group_rate):
    from utils import img_v_h_paste
    label_lists = []
    batch_size = 1
    group_img = np.zeros((len(group_list_dict), img_size, img_size, 3), dtype=np.uint8)
    group_label = np.zeros((len(group_list_dict), 1), dtype=np.int)

    # for fear of reload the same graph, we first reset graph into null
    keras.backend.clear_session()
    model_new = resnet101_model_new(224, 224, 3, 2)
    model_new.load_weights(model_path)

    inp = model_new.layers[0].input   # input placeholder
    assert len(inp.shape) == 4 
    input_shape_list = inp.shape.as_list()
    input_shape = (batch_size,input_shape_list[1],input_shape_list[2],input_shape_list[3])
    # reslayers = ['res4b22_relu','res4b22_branch2c','res5c','res5c_relu','avg_pool',
    # 'flatten_1','fc8']
    visual_layer = 'flatten_1' #batch_size*2048
    outputs = [layer.output for layer in model_new.layers if
               visual_layer in layer.name]
    # outputs = [layer.output for layer in model_new.layers]          # all layer outputs
    functors = [K.function([inp]+ [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

    #calculate the res_features
    features = [np.zeros((batch_size,2048),dtype=float)]

    
    for group in range(len(group_list_dict)):
        paths = []
        for index in range(len(group_list_dict[group])): 
            paths.append(group_list_dict[group][index]['file_path'])
        group_img = img_v_h_paste(paths,group_rate).resize((img_size,img_size),Image.ANTIALIAS)
        group_img_RGB_resize= group_img.resize((224,224),Image.ANTIALIAS)
        group_img_resize = np.asarray(group_img_RGB_resize,'f')#change to numpy
        img_preprocess = preprocess_input(group_img_resize)
        label_lists.append(group_list_dict[group][0]['label']) #random select a label 
        if group == 0:
            features = np.asarray(functors[0]([np.expand_dims(img_preprocess, axis=0), 0.])[0])
        else:
            features = np.append(features,np.asarray(functors[0]([np.expand_dims(img_preprocess, axis=0), 0.])[0]),axis=0)
        print ("interation:",group,"len(features):",int(len(features)))

    group_label = np.array(label_lists)
    print (visual_layer, features.shape)
    return features,group_label

def extract_pct_data(previous_json, current_json, X, load_select_txt_path, save_select_txt_path):
    selected_pre, unselected_pre, mask_pre = load_select_data(os.path.join(load_select_txt_path,previous_json))
    selected_cur, unselected_cur, mask_cur = load_select_data(os.path.join(load_select_txt_path,current_json))
    selected_pct = [i for i in selected_cur if i not in selected_pre]
    unselected_pct = [j for j in X if j not in selected_pct]
    mask_pct = [1 for n in range(0, len(X))]
    for i in range(len(X)):
        if i in selected_pct:
            mask_pct[i] = 0
    save_pct_select_data(save_select_txt_path, len(selected_cur), len(selected_pre), selected_pct,unselected_pct,mask_pct)

def get_img_label_from_group_list(group_list_dict,img_size,group_rate):
    # assert list(group_list_dict[0].keys()) == ['label','file_path']

    #ly
    from utils import img_v_h_paste
    group_img = np.zeros((len(group_list_dict), img_size, img_size, 3), dtype=np.uint8)
    group_label = np.zeros((len(group_list_dict), 1), dtype=np.int)
    imagefile = []
    for group in range(len(group_list_dict)):
        paths = []
        for index in range(len(group_list_dict[group])): 
            paths.append(group_list_dict[group][index]['file_path'])
        group_img[group] = img_v_h_paste(paths,group_rate).resize((img_size,img_size),Image.ANTIALIAS)
        group_label[group] = group_list_dict[group][0]['label']#random select a label  
    return group_img, group_label

def transform_group_data_label_list(data_label_lists, label):
    # for group_list_dict usage
    assert len(data_label_lists[0]) == 4
    trans_data_label_lists = data_label_lists
    for group in range(len(trans_data_label_lists)):
        paths = []
        for index in range(len(trans_data_label_lists[group])): 
            trans_data_label_lists[group][index]['label'] = label
    return trans_data_label_lists

def transform_data_label_list(data_label_lists, label):
    # for group_list_dict usage
    assert len(data_label_lists[0]) == 2
    trans_data_label_lists = data_label_lists
    for index in range(len(trans_data_label_lists)): 
        trans_data_label_lists[index]['label'] = label
    return trans_data_label_lists

def random_select_data(X,N,selected,unselected,mask):
    '''
    parameter
    X:input sorted index list[0,1,2,3,4,5,6]
    N: how many samples to select
    selected:list that are aready selected in X
    unselected:list that are not selected in X
    mask:list using 0/1 to represent each position is selected/unavaliable(0) or unselected/avaliable(1)
    write by @michelle
    '''
    # pdb.set_trace()
    new_samples = random.sample(unselected,N)
    for i in range(len(X)):
        if i in new_samples:
            mask[i] = 0
    selected.extend(new_samples)
    selected.sort()
    unselected = [j for j in X if j not in selected]
    print ("selected:",selected)
    print ("unselected:",unselected)
    print ("mask:",mask)
    return selected,unselected,mask

def uncertainty_select_data(model_path,X,N,selected,unselected,mask):
    pred_val_dicts = val_for_selection(model_path, unselected)
    # sorted_by_value_list = sorted(pred_val_dicts.items(), key=lambda kv: kv[1]) #larger one in the end
    from operator import itemgetter
    keys = []
    values = []
    #ranking probobility, low score in the front, high in the back
    for key, value in sorted(pred_val_dicts.items(), key = itemgetter(1), reverse = False):
        keys.append(int(key))
        values.append(float(value))
    # pdb.set_trace()
    new_samples = keys[0:N-1]#pick the low probobility
    selected.extend(new_samples)
    selected.sort()
    unselected = [j for j in X if j not in selected]
    for i in range(len(X)):
        if i in new_samples:
            mask[i] = 0
    print ("selected:",selected)
    print ("unselected:",unselected)
    print ("mask:",mask)
    # pdb.set_trace()
    return selected,unselected,mask
def uncertainty_select_data_for_fun(model_path,X,N,selected,unselected,mask):
    pred_val_dicts = val_for_selection_for_fun(model_path, unselected)
    # sorted_by_value_list = sorted(pred_val_dicts.items(), key=lambda kv: kv[1]) #larger one in the end
    from operator import itemgetter
    keys = []
    values = []
    #ranking probobility, low score in the front, high in the back
    for key, value in sorted(pred_val_dicts.items(), key = itemgetter(1), reverse = False):
        keys.append(int(key))
        # values.append(float(value))
        values.append(value)

    new_samples = keys[0:N-1]#pick the low probobility
    selected.extend(new_samples)
    selected.sort()
    unselected = [j for j in X if j not in selected]
    for i in range(len(X)):
        if i in new_samples:
            mask[i] = 0
    print ("selected:",selected)
    print ("unselected:",unselected)
    print ("mask:",mask)
    # pdb.set_trace()
    return selected,unselected,mask,pred_val_dicts,keys,values

def ud_select_data(model_path,X,N,selected,unselected,mask,ud_ratio):
    N_Unc = int(N * ud_ratio)
    N_Div = N - N_Unc

    #uncertainty pick(num:N_Unc)
    pred_val_dicts = val_for_selection(model_path, unselected)
    from operator import itemgetter
    keys = []
    values = []
    #ranking probobility, low score in the front, high in the back
    for key, value in sorted(pred_val_dicts.items(), key = itemgetter(1), reverse = False):
        keys.append(int(key))
        values.append(float(value))
    # pick
    new_samples = keys[0:N_Unc-1]#pick the low probobility
    # update selected/unselected/mask
    selected.extend(new_samples)
    selected.sort()
    unselected = [j for j in X if j not in selected]
    for i in range(len(X)):
        if i in new_samples:
            mask[i] = 0

    #diversity pick(num:N_Div) PCA + LSH into 20 clusters
    assert N_Div < len(unselected)
    new_samples = []
    buckect_dicts = val_for_selection_d(unselected)
    # circle pick one from each buckect till end
    circle_upperbound = len(buckect_dicts)
    circle = 0
    query_index = 0
    while query_index<N_Div:
        #control the index of the buckect_dicts[circle in 0-19]
        if circle == circle_upperbound:
            circle = 0
        # take one bucket and check if it is empty
        bucket_list = list(buckect_dicts[circle])
        if len(bucket_list) == 0:
            circle+=1
            continue
        # if it is not empty, pick one
        random.shuffle(bucket_list)
        name,index = bucket_list[0].split('_')
        new_samples.append(int(index))
        buckect_dicts[circle].remove(bucket_list[0])
        # if take one sample, query_index=query_index+1(pick in the next bucket)
        circle+=1
        query_index+=1

    # update selected/unselected/mask
    selected.extend(new_samples)
    selected.sort()
    unselected = [j for j in X if j not in selected]
    for i in range(len(X)):
        if i in new_samples:
            mask[i] = 0

    # final result    
    print ("selected:",selected)
    print ("unselected:",unselected)
    print ("mask:",mask)
    # pdb.set_trace()
    return selected,unselected,mask

def group_data(group_rate,sequence):
    '''
    group data for sequence
    if len(sequence)%each_group_num !=0, we will fill up for group[len(sequence)%each_group_num+1]
    randomly selected in original sequence
    '''
    import math 
    group_list = []
    each_group_num = group_rate * group_rate
    select_time = math.ceil(len(sequence)/each_group_num) #xiang shang qu zheng
    all_sequence = sequence

    for i in range(select_time):
        temp_sequence = sequence
        if len(temp_sequence) < each_group_num:
            #if the last selection is not enough sample to select, select from all_sequence for making up
            rest_data_length = each_group_num-len(temp_sequence)
            last = temp_sequence
            last.extend(random.sample(all_sequence,rest_data_length))
            group_list.append(last)
        else:
            group_list.append(random.sample(temp_sequence,each_group_num))
        sequence = [j for j in temp_sequence if j not in group_list[i]]
    return group_list

def group_data_withsame(group_rate,sequence):
    '''
    group data for sequence
    if len(sequence)%each_group_num !=0, we will fill up for group[len(sequence)%each_group_num+1]
    randomly selected in original sequence
    '''
    import math 
    group_list = []
    each_group_num = group_rate * group_rate
    select_time = len(sequence) #paste the same pic in each_group_num as a single pic

    for i in range(select_time):
        single_list = []
        for j in range(each_group_num):# group each_group_num single data
            single_list.append(sequence[i])
        group_list.append(single_list)
    return group_list

def group_data_sequencial(group_rate,sequence):
    '''
    group data for sequence
    if len(sequence)%each_group_num !=0, we will fill up for group[len(sequence)%each_group_num+1]
    randomly selected in original sequence
    '''
    import math 
    group_list = []
    each_group_num = group_rate * group_rate
    select_time = math.ceil(len(sequence)/each_group_num) #xiang shang qu zheng
    all_sequence = sequence

    for i in range(select_time):
        temp_sequence = sequence
        if len(temp_sequence) < each_group_num:
            #if the last selection is not enough sample to select, select from all_sequence for making up
            rest_data_length = each_group_num-len(temp_sequence)
            last = temp_sequence
            last.extend(random.sample(all_sequence,rest_data_length))
            group_list.append(last)
        else:
            group_list.append(sequence[0: each_group_num])
        sequence = [j for j in temp_sequence if j not in group_list[i]]
    return group_list

def group_data_frontback(group_rate,sequence):
    '''
    group data for sequence
    if len(sequence)%each_group_num !=0, we will fill up for group[len(sequence)%each_group_num+1]
    randomly selected in original sequence
    '''
    import math 
    import random
    group_list = []
    each_group_num = group_rate * group_rate
    select_time = math.ceil(len(sequence)/each_group_num) #xiang shang qu zheng
    all_sequence = sequence

    for i in range(select_time):
        temp_sequence = sequence
        single_list = []
        if len(temp_sequence) < each_group_num:
            #if the last selection is not enough sample to select, select from all_sequence for making up
            rest_data_length = each_group_num-len(temp_sequence)
            last = temp_sequence
            last.extend(random.sample(all_sequence,rest_data_length))
            group_list.append(last)
        else:
            for j in range(group_rate):
                single_list.append(sequence[j])
            for k in range(group_rate):
                single_list.append(sequence[len(sequence)-k-1])
            random.shuffle(single_list)
            group_list.append(single_list)
        sequence = [j for j in temp_sequence if j not in group_list[i]]
    return group_list

def load_select_data(filename):
    import json
    with open(filename) as f:
        data = json.load(f)
        assert list(data.keys()) == ['selected', 'unselected', 'mask']
    return data['selected'],data['unselected'],data['mask']

def save_select_data(save_select_txt_path,selected,unselected,mask):
    import json
    dictObj = {'selected':selected,'unselected':unselected,'mask':mask}
    jsObj = json.dumps(dictObj)
    save_name = os.path.join(save_select_txt_path,str(len(selected))+ '_' + currtime + '.json')
    fileObject = open(save_name, 'w')
    fileObject.write(jsObj)
    fileObject.close()

def save_pct_select_data(save_select_txt_path,pre_num,cur_num,selected,unselected,mask):
    import json
    dictObj = {'selected':selected,'unselected':unselected,'mask':mask}
    jsObj = json.dumps(dictObj)
    save_name = os.path.join(save_select_txt_path,str(cur_num) + '_' + str(pre_num) + '_' + 
                            str(len(selected))+ '_' + currtime + '.json')
    fileObject = open(save_name, 'w')
    fileObject.write(jsObj)
    fileObject.close()

'''
'''