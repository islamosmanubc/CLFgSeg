#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import random as rn
import os,sys
from keras.preprocessing.image import save_img

# set current working directory
cur_dir = os.getcwd()
os.chdir(cur_dir)
sys.path.append(cur_dir)

# =============================================================================
#  For reprodocable results, from keras.io
# =============================================================================
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
from keras import backend as K
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

import keras, glob
from keras.preprocessing import image as kImage
from Resnet_CDMLGen import ResSegModel
#from Resnet_CDML import ResSegModel
from keras.utils.data_utils import get_file
import gc

# alert the user
if keras.__version__!= '2.0.6' or tf.__version__!='1.1.0' or sys.version_info[0]<3:
    print('We implemented using [keras v2.0.6, tensorflow-gpu v1.1.0, python v3.6.3], other versions than these may cause errors somehow!\n')

# Few training frames, it may fit in memory
def getfiles(dataset):
    
    Y_list = []
    X_list = []
    E_list = []


    if dataset == 'offline':
        pathf = 'D:/phd/my_papers/paper7(LL)/MyCode/datasets/offline/train'
        tasks = os.listdir(pathf)
        for task in tasks:
            scenes = os.listdir(pathf+'/'+task+'/Annotations')
            for scene in scenes:
                Y_list.append(glob.glob(os.path.join(pathf,task,'Annotations', scene, '*.png')))
                X_list.append(glob.glob(os.path.join(pathf,task,'JPEGImages', scene,'*.jpg')))
                E_list.append(glob.glob(os.path.join(pathf,task,'Edges', scene,'*.png')))

    basepath = 'C:/Users/islam/source/repos/PythonApplication1/PythonApplication1/'
    if dataset == 'base':
        pathf = basepath+'datasets/train/JPEGImages/base_foreground_segmentation/yovos'
        tasks = os.listdir(pathf)
        pathf = basepath+'datasets/train/'
        for task in tasks:
            Y_list.append(glob.glob(os.path.join(pathf,'Annotations/base_foreground_segmentation/yovos', task, '*.png')))
            X_list.append(glob.glob(os.path.join(pathf,'JPEGImages/base_foreground_segmentation/yovos', task,'*.jpg')))
            E_list.append(glob.glob(os.path.join(pathf,'Annotations/base_foreground_segmentation/yovos', task,'*.png')))

    elif dataset == 'segtrack':
        pathf = 'datasets/segtrackv212/JPEGImages1'
        tasks = os.listdir(pathf)
        for task in tasks:
            Y_list.append(glob.glob(os.path.join('datasets/segtrackv212/GroundTruth',task, '*.png')))
            X_list.append(glob.glob(os.path.join('datasets/segtrackv212/JPEGImages1',task, 'input','*.png')))
            E_list.append(glob.glob(os.path.join('datasets/segtrackv212/GroundTruth',task, 'edge','*.png')))


        for k in range(len(Y_list)):
           Y_list_temp = []
           E_list_temp = []
           for i in range(len(X_list[k])):
               X_name = os.path.basename(X_list[k][i])
               X_name = X_name.split('.')[0]
               
               for j in range(len(Y_list[k])):
                   Y_name = os.path.basename(Y_list[k][j])
                   Y_name = Y_name.split('.')[0]
                   if (Y_name == X_name):
                       Y_list_temp.append(Y_list[k][j])
                       break

               for j in range(len(E_list[k])):
                   E_name = os.path.basename(E_list[k][j])
                   E_name = E_name.split('.')[0]
                   if (E_name == X_name):
                       E_list_temp.append(E_list[k][j])
                       break

           Y_list[k] = Y_list_temp
           E_list[k] = E_list_temp

    xlist = []
    ylist = []
    elist = []
    for k in range(len(X_list)):
        for i in range(len(Y_list[k])):
            xlist.append(X_list[k][i])
            ylist.append(Y_list[k][i])
            elist.append(E_list[k][i])    
            
    
    X_list = xlist
    Y_list = ylist
    E_list = elist
    
    X_list = np.array(X_list)
    Y_list = np.array(Y_list)
    E_list = np.array(E_list)
    idx = list(range(X_list.shape[0]))
    np.random.shuffle(idx)
    np.random.shuffle(idx)
    X_list = X_list[idx]
    Y_list = Y_list[idx]
    E_list = E_list[idx]

    return X_list,Y_list,E_list

def getImgs(X_list,Y_list,E_list):

    # load training data
    num_imgs = len(X_list)
    X = np.zeros((num_imgs,240,320,3),dtype="float32")
    Y = np.zeros((num_imgs,240,320,2),dtype="float32")
    for i in range(len(X_list)):
        x = kImage.load_img(X_list[i],target_size = [240,320,3])
        x = kImage.img_to_array(x)
        X[i,:,:,:] = x
        
        x = kImage.load_img(Y_list[i], grayscale = True,target_size = [240,320])
        x = kImage.img_to_array(x)
        x /= 255.0
        x = np.floor(x)
        Y[i,:,:,0] = np.reshape(x,(240,320))
        
        x = kImage.load_img(E_list[i], grayscale = True,target_size = [240,320])
        x = kImage.img_to_array(x)
        x /= 255.0
        x = np.floor(x)
        Y[i,:,:,1] = np.reshape(x,(240,320))
        
        
    idx = list(range(X.shape[0]))
    np.random.shuffle(idx)
    np.random.shuffle(idx)
    X = X[idx]
    Y = Y[idx]

    
    return X, Y
### training function    
def train(X, Y, E,data_name):
    
    lr = 1e-4
    max_epoch = 20
    batch_size = 500
    #batch_size = 10
    dd = len(X)
    ###
    
    
    model = ResSegModel(lr, (240,320,3), 'x')
    model = model.initModel('CDnet')
    
    save_sample_path='D:/paper6_results'


    sub_dataset = int(len(X)/batch_size)
    for epoch in range(max_epoch):
        print("\nStart of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.
        for step in range(batch_size):
            #y = Y[step*sub_dataset:(step+1)*sub_dataset,:,:,:]
            #x = X[step*sub_dataset:(step+1)*sub_dataset,:,:,:]
            
            y = Y[step*sub_dataset:(step+1)*sub_dataset]
            x = X[step*sub_dataset:(step+1)*sub_dataset]
            e = E[step*sub_dataset:(step+1)*sub_dataset]
            
            cx,cy = getImgs(x,y,e)
            cy = cy[:,:,:,:1]
            #[x,o4,o3,o2,o1]
            model.fit(cx, [cy,cy,cy,cy,cy],
                epochs=1, batch_size=1, verbose=2, shuffle = False)

        weights = model.get_weights()
        a = np.array(weights)
        np.save('last_weights/pre/'+data_name+'_weights_refnet_pre'+str(epoch)+'.npy', a)
            
    del model


# =============================================================================
# Main func
# =============================================================================



x,y,e = getfiles('base')
train(x,y,e,'base')
gc.collect()