#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import random as rn
import os,sys

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
from sklearn.utils import compute_class_weight
from Resnet_CDMLGen_CL_N import ResSegModel
#from Resnet_CD import ResSegModel
from sklearn.metrics import f1_score
from keras.utils.data_utils import get_file
import gc
import time
# alert the user
if keras.__version__!= '2.0.6' or tf.__version__!='1.1.0' or sys.version_info[0]<3:
    print('We implemented using [keras v2.0.6, tensorflow-gpu v1.1.0, python v3.6.3], other versions than these may cause errors somehow!\n')
def closed_range(start, stop, step=1):
  dir = 1 if (step > 0) else -1
  return range(start, stop-1 + dir, step)
# Few training frames, it may fit in memory
def getfiles(dataset,task):
    
    Y_list = []
    X_list = []
    E_list = []

    task = str(task)
    if dataset == 'offline':
        pathf = 'D:/phd/my_papers/paper7(LL)/MyCode/datasets/offline/train'
        tasks = os.listdir(pathf)
        
        scenes = os.listdir(pathf+'/'+task+'/Annotations')
        for scene in scenes:
            Y_list.append(glob.glob(os.path.join(pathf,task,'Annotations', scene, '*.png')))
            X_list.append(glob.glob(os.path.join(pathf,task,'JPEGImages', scene,'*.jpg')))
            E_list.append(glob.glob(os.path.join(pathf,task,'Edges', scene,'*.png')))

    basepath = 'C:/Users/islam/source/repos/PythonApplication1/PythonApplication1/'
    if dataset == 'base':
        pathf = basepath+'datasets/test/'
        Y_list.append(glob.glob(os.path.join(pathf,'Annotations/base_foreground_segmentation/yovos', task, '*.png')))
        X_list.append(glob.glob(os.path.join(pathf,'JPEGImages/base_foreground_segmentation/yovos', task,'*.jpg')))
        E_list.append(glob.glob(os.path.join(pathf,'Annotations/base_foreground_segmentation/yovos', task,'*.png')))
    basepath = 'C:/Users/islam/source/repos/PythonApplication1/PythonApplication1/'
    if dataset == 'continual':
        pathf = basepath+'datasets/test/'
        Y_list.append(glob.glob(os.path.join(pathf,'Annotations/continual_foreground_segmentation/cdnet', task, '*.png')))
        X_list.append(glob.glob(os.path.join(pathf,'JPEGImages/continual_foreground_segmentation/cdnet', task,'*.jpg')))
        E_list.append(glob.glob(os.path.join(pathf,'Annotations/continual_foreground_segmentation/cdnet', task,'*.png')))
    if dataset == 'fewshot':
        pathf = basepath+'datasets/test/'
        Y_list.append(glob.glob(os.path.join(pathf,'Annotations/fewshot_foreground_segmentation/davis', task, '*.png')))
        X_list.append(glob.glob(os.path.join(pathf,'JPEGImages/fewshot_foreground_segmentation/davis', task,'*.jpg')))
        E_list.append(glob.glob(os.path.join(pathf,'Annotations/fewshot_foreground_segmentation/davis', task,'*.png')))
    if dataset == 'online':
        pathf = 'D:/phd/my_papers/paper7(LL)/MyCode/datasets/online/train'
        tasks = os.listdir(pathf)
        
        scenes = os.listdir(pathf+'/'+str(task)+'/Annotations')
        for scene in scenes:
            Y_list.append(glob.glob(os.path.join(pathf,task,'Annotations', scene, '*.png')))
            X_list.append(glob.glob(os.path.join(pathf,task,'JPEGImages', scene,'*.jpg')))
            E_list.append(glob.glob(os.path.join(pathf,task,'Edges', scene,'*.png')))
    

    if dataset == 'few':
        pathf = 'D:/phd/my_papers/paper7(LL)/MyCode/datasets/few/train_5shot'
        tasks = os.listdir(pathf)
        
        scenes = os.listdir(pathf+'/'+str(task)+'/Annotations')
        for scene in scenes:
            Y_list.append(glob.glob(os.path.join(pathf,task,'Annotations', scene, '*.png')))
            X_list.append(glob.glob(os.path.join(pathf,task,'JPEGImages', scene,'*.jpg')))
            E_list.append(glob.glob(os.path.join(pathf,task,'Edges', scene,'*.png')))
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
            #elist.append(E_list[k][i])    
            
    
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
    #E_list = E_list[idx]

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
        
        #x = kImage.load_img(E_list[i], grayscale = True,target_size = [240,320])
        #x = kImage.img_to_array(x)
        #x /= 255.0
        #x = np.floor(x)
        Y[i,:,:,1] = np.reshape(x,(240,320))
        
        
    idx = list(range(X.shape[0]))
    np.random.shuffle(idx)
    np.random.shuffle(idx)
    X = X[idx]
    Y = Y[idx]

    
    return X, Y

### training function    
def test(model,X, Y, E,data_name,task,f,ths,fm_all,iou_all,exp):
    
    ### hyper-params
    max_epoch = 5
    dd = len(X)
    ###
    
    epsilon = 1e-6
    
    tasks = 10  
    lr = 1e-4
    
    start = time.time()
    weights = np.load('last_weights/weights_tbpi/'+data_name+exp+'_weights_refnet_tbpi_'+task+'.npy', allow_pickle=True, encoding="latin1").tolist()
    start = time.time()
    #weights = np.load('weights_pretrain.npy', allow_pickle=True, encoding="latin1").tolist()
    #a = np.array(model.get_weights())
    model.set_weights(weights)

    cx,cy = getImgs(X,Y,E)
    cy = cy[:,:,:,:1]

    
    yhat = model.predict(cx[0:1,:,:,:])[0]
    end = time.time()
    print(f"initialization time = {end-start}")

    #yhat = np.zeros((len(cx),240,320,1))
    #for i in range(len(cx)):
    start = time.time()

    yhat = model.predict(cx)[0]
    
    end = time.time()
    print(f"the runtime is {end-start} and the len = {len(cx)}\n")
    
    #yhat[i,:,:,:]=yy
    #yhat=yy
    
    actualy = cy
    max_iou = 0
    max_th = 0
    max_fscore = 0
    yp = np.zeros((len(yhat),240,320,1))
    


    iou_t = {}
    fm_t = {}
    for th in ths:
        iou_t.update({th:[]})
        fm_t.update({th:[]})

    for th in ths:
        yp[yhat>=th]=1
        yp[yhat<th]=0

        pred = (yp==1)
        gt = (actualy==1)

        pred = np.reshape(pred,(len(yhat),240,320))
        gt = np.reshape(gt,(len(yhat),240,320))

        gtflat = gt.flatten()
        predflat = pred.flatten()

        fscore = f1_score(gtflat,predflat,average='binary')

        intersection = np.sum((pred*gt),axis=1)
        intersection = np.sum(intersection,axis=1)
        union = np.sum((pred+gt),axis=1)
        union = np.sum(union,axis=1)
        iou = np.mean((intersection+epsilon)/(union+epsilon))
        iou_t[th].append(iou)
        fm_t[th].append(fscore)
        iou_all[th].append(iou)
        fm_all[th].append(fscore)
        if iou > max_iou:
            max_iou = iou
            max_th = th
            max_fscore = fscore


        f.write('Threshold = '+str(th)+'\t')
        f.write('iou = '+str(np.mean(np.array(iou_t[th])))+'\t')
        f.write('fm = '+str(np.mean(np.array(fm_t[th])))+'\n')
        f.write('========================\n')


# =============================================================================
# Main func
# =============================================================================

def mainfunc(Tasks):
    ths = [0.2,0.4,0.6,0.8]
    iou_all ={}
    fm_all = {}
    for th in ths:
        iou_all.update({th:[]})
        fm_all.update({th:[]})
    
    tasks = 10  
    lr = 1e-4
    model = ResSegModel(lr, (240,320,3), '',tasks)
    model = model.initModel('CDnet')
    model.summary()
    tasks = Tasks
    # =============================================================================
    num_frames = 200 # either 25 or 200 training frames
    # =============================================================================
    basepath = 'C:/Users/islam/source/repos/PythonApplication1/PythonApplication1/'
    pathf = basepath+'datasets/train/JPEGImages/base_foreground_segmentation/yovos'
    pathf = basepath+'datasets/train/JPEGImages/continual_foreground_segmentation/cdnet'
    pathf = basepath+'datasets/train/JPEGImages/fewshot_foreground_segmentation/davis'
    videos = os.listdir(pathf)
    
    T = tasks
    ee=0

    dataset = 'continual'
    dataset = 'fewshot'
    exp = '_5s'
    f = open(dataset+exp+'_results.txt','a')

    for v in videos:
        print ('Training ->>> task ' + v)
        dataname = dataset
        x,y,e = getfiles(dataname,v)
        test(model,x,y,e,dataname,v,f,ths,fm_all,iou_all,exp)
        gc.collect()
    f.write('All dataset: \n')
    for th in ths:
        iou = np.mean(np.array(iou_all[th]))
        fm = np.mean(np.array(fm_all[th]))
        f.write('Threshold = '+str(th)+'\t')
        f.write('iou = '+str(iou)+'\t')
        f.write('fm = '+str(fm)+'\n')
    f.close()


#mainfunc(80)  
mainfunc(10)