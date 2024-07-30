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
from keras.utils.data_utils import get_file
import gc

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
        pathf = basepath+'datasets/train/'
        Y_list.append(glob.glob(os.path.join(pathf,'Annotations/base_foreground_segmentation/yovos', task, '*.png')))
        X_list.append(glob.glob(os.path.join(pathf,'JPEGImages/base_foreground_segmentation/yovos', task,'*.jpg')))
        E_list.append(glob.glob(os.path.join(pathf,'Annotations/base_foreground_segmentation/yovos', task,'*.png')))
    basepath = 'C:/Users/islam/source/repos/PythonApplication1/PythonApplication1/'
    if dataset == 'continual':
        pathf = basepath+'datasets/train/'
        Y_list.append(glob.glob(os.path.join(pathf,'Annotations/continual_foreground_segmentation/cdnet', task, '*.png')))
        X_list.append(glob.glob(os.path.join(pathf,'JPEGImages/continual_foreground_segmentation/cdnet', task,'*.jpg')))
        E_list.append(glob.glob(os.path.join(pathf,'Annotations/continual_foreground_segmentation/cdnet', task,'*.png')))
    if dataset == 'fewshot':
        pathf = basepath+'datasets/train/'
        Y_list.append(glob.glob(os.path.join(pathf,'Annotations/fewshot_foreground_segmentation/davis_5shot', task, '*.png')))
        X_list.append(glob.glob(os.path.join(pathf,'JPEGImages/fewshot_foreground_segmentation/davis_5shot', task,'*.jpg')))
        E_list.append(glob.glob(os.path.join(pathf,'Annotations/fewshot_foreground_segmentation/davis_5shot', task,'*.png')))
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
def train(model,weights,X, Y, E,data_name,task):
    
    ### hyper-params
    max_epoch = 5
    dd = len(X)
    ###
    
    
    tasks = 10  
    lr = 1e-4
    
    #weights = np.load('weights_pretrain.npy', allow_pickle=True, encoding="latin1").tolist()
    a = np.array(model.get_weights())
    
    tp = int(128/tasks)
    ntp = 128-tp
    a[0] = weights[0][:,:,:,:ntp]
    a[1] = weights[1][:ntp]
    a[2] = weights[0][:,:,:,ntp:]
    a[3] = weights[1][ntp:]

    a[4] = weights[2][:,:,:,:ntp]
    a[5] = weights[3][:ntp]
    a[6] = weights[2][:,:,:,ntp:]
    a[7] = weights[3][ntp:]

    a[8] = weights[4][:,:,:,:ntp]
    a[9] = weights[5][:ntp]
    a[10] = weights[4][:,:,:,ntp:]
    a[11] = weights[5][ntp:]

    a[12] = weights[6][:,:,:,:ntp]
    a[13] = weights[7][:ntp]
    a[14] = weights[6][:,:,:,ntp:]
    a[15] = weights[7][ntp:]
    ######
    tp = int(128/tasks)
    ntp = 128-tp
    a[16] = weights[8][:,:,:,:ntp]
    a[17] = weights[9][:ntp]
    a[18] = weights[8][:,:,:,ntp:]
    a[19] = weights[9][ntp:]

    a[20] = weights[10][:,:,:,:ntp]
    a[21] = weights[11][:ntp]
    a[22] = weights[10][:,:,:,ntp:]
    a[23] = weights[11][ntp:]

    a[24] = weights[12][:,:,:,:ntp]
    a[25] = weights[13][:ntp]
    a[26] = weights[12][:,:,:,ntp:]
    a[27] = weights[13][ntp:]

    a[28] = weights[14][:,:,:,:ntp]
    a[29] = weights[15][:ntp]
    a[30] = weights[14][:,:,:,ntp:]
    a[31] = weights[15][ntp:]

    ####
    tp = int(128/tasks)
    ntp = 128-tp
    a[32] = weights[16][:,:,:,:ntp]
    a[33] = weights[17][:ntp]
    a[34] = weights[16][:,:,:,ntp:]
    a[35] = weights[17][ntp:]

    a[36] = weights[18][:,:,:,:ntp]
    a[37] = weights[19][:ntp]
    a[38] = weights[18][:,:,:,ntp:]
    a[39] = weights[19][ntp:]

    a[40] = weights[20][:,:,:,:ntp]
    a[41] = weights[21][:ntp]
    a[42] = weights[20][:,:,:,ntp:]
    a[43] = weights[21][ntp:]

    a[44] = weights[22][:,:,:,:ntp]
    a[45] = weights[23][:ntp]
    a[46] = weights[22][:,:,:,ntp:]
    a[47] = weights[23][ntp:]
    ######
    tp = int(128/tasks)
    ntp = 128-tp
    a[48] = weights[24][:,:,:,:ntp]
    a[49] = weights[25][:ntp]
    a[50] = weights[24][:,:,:,ntp:]
    a[51] = weights[25][ntp:]

    a[52] = weights[26][:,:,:,:ntp]
    a[53] = weights[27][:ntp]
    a[54] = weights[26][:,:,:,ntp:]
    a[55] = weights[27][ntp:]

    a[56] = weights[28][:,:,:,:ntp]
    a[57] = weights[29][:ntp]
    a[58] = weights[28][:,:,:,ntp:]
    a[59] = weights[29][ntp:]

    a[60] = weights[30][:,:,:,:ntp]
    a[61] = weights[31][:ntp]
    a[62] = weights[30][:,:,:,ntp:]
    a[63] = weights[31][ntp:]

    ####
    tp = int(128/tasks)
    ntp = 128-tp
    a[64] = weights[32][:,:,:,:ntp]
    a[65] = weights[33][:ntp]
    a[66] = weights[32][:,:,:,ntp:]
    a[67] = weights[33][ntp:]

    a[68] = weights[34][:,:,:,:ntp]
    a[69] = weights[35][:ntp]
    a[70] = weights[34][:,:,:,ntp:]
    a[71] = weights[35][ntp:]

    a[72] = weights[36][:,:,:,:ntp]
    a[73] = weights[37][:ntp]
    a[74] = weights[36][:,:,:,ntp:]
    a[75] = weights[37][ntp:]

    a[76] = weights[38][:,:,:,:ntp]
    a[77] = weights[39][:ntp]
    a[78] = weights[38][:,:,:,ntp:]
    a[79] = weights[39][ntp:]
    
    a[80] = weights[40][:,:,:,:ntp]
    a[81] = weights[41][:ntp]
    a[82] = weights[40][:,:,:,ntp:]
    a[83] = weights[41][ntp:]

    a[84] = weights[42]
    a[85] = weights[43]

    a[86] = weights[44][:,:,:,:ntp]
    a[87] = weights[45][:ntp]
    a[88] = weights[44][:,:,:,ntp:]
    a[89] = weights[45][ntp:]

    a[94] = weights[46]
    a[95] = weights[47]

    a[90] = weights[48][:,:,:,:ntp]
    a[91] = weights[49][:ntp]
    a[92] = weights[48][:,:,:,ntp:]
    a[93] = weights[49][ntp:]
    
    tp = int(128/tasks)
    ntp = 128-tp
    a[96] = weights[50][:,:,:,:ntp]
    a[97] = weights[51][:ntp]
    a[98] = weights[50][:,:,:,ntp:]
    a[99] = weights[51][ntp:]
    
    a[100] = weights[52]
    a[101] = weights[53]

    tp = int(128/tasks)
    ntp = 128-tp
    a[102] = weights[54][:,:,:,:ntp]
    a[103] = weights[55][:ntp]
    a[104] = weights[54][:,:,:,ntp:]
    a[105] = weights[55][ntp:]
    
    a[106] = weights[56]
    a[107] = weights[57]
    a[108] = weights[58]
    a[109] = weights[59]

    a[110] = weights[60]
    a[111] = weights[61]
    a[112] = weights[62]
    a[113] = weights[63]

    
    tp = int(128/tasks)
    ntp = 128-tp
    a[114] = weights[64][:,:,:,:ntp]
    a[115] = weights[65][:ntp]
    a[116] = weights[64][:,:,:,ntp:]
    a[117] = weights[65][ntp:]

    
    a[118] = weights[66][:,:,:,:ntp]
    a[119] = weights[67][:ntp]
    a[120] = weights[66][:,:,:,ntp:]
    a[121] = weights[67][ntp:]

    
    a[122] = weights[68][:,:,:,:ntp]
    a[123] = weights[69][:ntp]
    a[116] = weights[68][:,:,:,ntp:]
    a[125] = weights[69][ntp:]

    
    a[126] = weights[70][:,:,:,:ntp]
    a[127] = weights[71][:ntp]
    a[128] = weights[70][:,:,:,ntp:]
    a[129] = weights[71][ntp:]

    a[130] = weights[72]
    a[131] = weights[73]

    a[132] = weights[74][:,:,:,:ntp]
    a[133] = weights[75][:ntp]
    a[134] = weights[74][:,:,:,ntp:]
    a[135] = weights[75][ntp:]

    a[136] = weights[76]
    a[137] = weights[77]
    
    model.set_weights(a)

    cx,cy = getImgs(X,Y,E)
    cy = cy[:,:,:,:1]
    model.fit(cx, [cy,cy,cy,cy,cy],
        epochs=max_epoch, batch_size=1, verbose=2, shuffle = False)
    
    a = np.array(model.get_weights())
    np.save('last_weights/weights_tbpi/'+data_name+'_5s_weights_refnet_tbpi_'+str(task)+'.npy', a)
    del model



# =============================================================================
# Main func
# =============================================================================

def mainfunc(Tasks):

    
    tasks = 10  
    lr = 1e-4
    model = ResSegModel(lr, (240,320,3), '',tasks)
    model = model.initModel('CDnet')
    
    weights = np.load('last_weights/pre/base_weights_refnet_pre10.npy', allow_pickle=True, encoding="latin1").tolist()
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
    for v in videos:
        print ('Training ->>> task ' + v)
        dataname = 'fewshot'
        x,y,e = getfiles(dataname,v)
        train(model,weights,x,y,e,dataname,v)
        gc.collect()
        #break


#mainfunc(80)  
mainfunc(10)