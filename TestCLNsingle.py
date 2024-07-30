

"""
Created on Mon Jun 27 2018

@author: longang
"""

# coding: utf-8
#get_ipython().magic(u'load_ext autotime')
import numpy as np
import os, glob, sys
from keras.preprocessing import image as kImage
from keras.preprocessing.image import save_img
from Resnet_CDMLGen_CL_N import ResSegModel
#from Resnet_CDMLGen import ResSegModel
#from skimage.transform import pyramid_gaussian
from keras.models import load_model
#from scipy.misc import imsave#, imresize
import gc


# Optimize to avoid memory exploding. 
# For each video sequence, we pick only 1000frames where res > 400
# You may modify according to your memory/cpu spec.
def checkFrame(X_list):
    img = kImage.load_img(X_list[0])
    img = kImage.img_to_array(img).shape # (480,720,3)
    num_frames = len(X_list) # 7000
    max_frames = 1000 # max frames to slice
    if(img[1]>=400 and len(X_list)>max_frames):
        print ('\t- Total Frames:' + str(num_frames))
        num_chunks = num_frames/max_frames
        num_chunks = int(np.ceil(num_chunks)) # 2.5 => 3 chunks
        start = 0
        end = max_frames
        m = [0]* num_chunks
        for i in range(num_chunks): # 5
            m[i] = range(start, end) # m[0,1500], m[1500, 3000], m[3000, 4500]
            start = end # 1500, 3000, 4500 
            if (num_frames - start > max_frames): # 1500, 500, 0
                end = start + max_frames # 3000
            else:
                end = start + (num_frames- start) # 2000 + 500, 2500+0
        print ('\t- Slice to:' + str(m))
        del img, X_list
        return [True, m]
    del img, X_list
    return [False, None]
    
# Load some frames (e.g. 1000) for segmentation
def generateData(scene_input_path, X_list,Z_list, scene,c):
    # read images
    
    T = np.zeros([len(X_list),240,320,6])
    print ('\n\t- Loading frames:')
    for i in range(0, len(X_list)):
        img = kImage.load_img(X_list[i],target_size=[240,320,3])
        x = kImage.img_to_array(img)
        
        if c == 'PTZ':
            img = kImage.load_img(Z_list[i],target_size=[240,320,3])
            z = kImage.img_to_array(img)
        else:
            img = kImage.load_img(Z_list[0],target_size=[240,320,3])
            z = kImage.img_to_array(img)

        T[i,:,:,:3] = x
        T[i,:,:,3:] = z

        sys.stdout.write('\b' * len(str(i)))
        sys.stdout.write('\r')
        sys.stdout.write(str(i+1))
    
    del img, x, X_list
    #X = np.asarray(X)
    print ('\nShape' + str(T.shape))

    return T 

def getFiles(scene_input_path):
    inlist = glob.glob(os.path.join(scene_input_path,'*.jpg'))
    return np.asarray(inlist)

def closed_range(start, stop, step=1):
  dir = 1 if (step > 0) else -1
  return range(start, stop-1 + dir, step)

def mainfunc(tasknum):
    sequences = [
                       'office',
                       'badminton',
                       'skating', 
                       'overpass',
                        'sofa',
                        'peopleInShade',
                       'diningRoom',
                       'library']


    categories = ['baseline', 
                       'cameraJitter',
                       'badWeather',
                       'dynamicBackground',
                        'intermittentObjectMotion',
                        'shadow',
                        'thermal',
                       'thermal',]
    # Extract all mask
    dataset = {
               'baseline':[
                       'office'
                       ],
               'cameraJitter':[
                       'badminton'
                       ],
               'badWeather':[
                       'skating'
                       ],
                'dynamicBackground':[
                        'overpass'
                        ],
                'intermittentObjectMotion':[
                        'sofa'
                        ],
                'shadow':[
                        'peopleInShade'
                        ],
               'thermal':[
                       'diningRoom',
                       'library'
                       ]
    }

    # number of exp frame (25, 50, 200)
    num_frames = 200

    # 1. Raw RGB frame to extract foreground masks, downloaded from changedetection.net
    raw_dataset_dir = 'D:/phd/moving_object_detection/code/FgSegNet_v2_master/datasets/CDnet2014_dataset'

    # 2. model dir

    # 3. path to store results
    #results_dir = os.path.join('ResSegNet', 'results4CL' + str(num_frames))
    fe = 4
    results_dir = os.path.join('ResSegNet', 'dec'+str(fe))
    #results_dir = os.path.join('ResSegNet', 'edge_results_original_size')

    ROI_file = 'D:/phd/moving_object_detection/code/FgSegNet_v2_master/datasets/CDnet2014_dataset/badWeather/blizzard/ROI.jpg'
    img = kImage.load_img(ROI_file, target_size=[240,320])
    img = kImage.img_to_array(img)
    img_shape = img.shape
    model = ResSegModel(0, (240,320,6), 'x',tasknum)
    #model = ResSegModel(0, (240,320,6), 'x')
    model = model.initModel('CDnet')


    # Loop through all categories (e.g. baseline)
    c=0
    Tasks = tasknum
    for t in range(Tasks):
    
        #weights = np.load('weights_dense_pretrain_edge_200_0.npy', allow_pickle=True, encoding="latin1").tolist()
        weights = np.load('weights_cl'+str(tasknum)+'_'+str(t)+'.npy', allow_pickle=True, encoding="latin1").tolist()
        a = np.array(model.get_weights())
        for k in range(len(weights)):
            a[k] = weights[k]
    
        model.set_weights(a)
        #b = np.array(model.get_weights())
        y = len(sequences)
        oney = int(y/Tasks)
        currentystart = oney*t
        currentyend = currentystart+oney
        if(t+1 == Tasks):
            currentyend = y

        csequences = []
        ccategories = []
        for i in closed_range(currentystart, currentyend):
            csequences.append(sequences[i])
            ccategories.append(categories[i])

        # Loop through all scenes (e.g. highway, ...)
        for q in range(len(csequences)):
            scene = csequences[q]
            category = ccategories[q]
            print ('\n->>> '+ category +' / ' + scene)
        


            mask_dir = os.path.join(results_dir, category, scene)
            if not os.path.exists(mask_dir):
                os.makedirs(mask_dir)

        

            # path of dataset downloaded from CDNet
            scene_input_path = os.path.join(raw_dataset_dir, category, scene, 'input')
            scene_input2_path = os.path.join(raw_dataset_dir, category, scene, 'back')
            # path of ROI to exclude non-ROI
            # make sure that each scene contains ROI.bmp and have the same dimension as raw RGB frames
            ROI_file = os.path.join(raw_dataset_dir, category, scene, 'ROI.bmp')
        
            # refer to http://jacarini.dinf.usherbrooke.ca/datasetOverview/
            img = kImage.load_img(ROI_file, grayscale=True,target_size=[240,320])
            img = kImage.img_to_array(img)
            img = img.reshape(-1) # to 1D
            idx = np.where(img == 0.)[0] # get the non-ROI, black area
            del img
        
            # load path of files
            X_list = getFiles(scene_input_path)
            fname = os.path.basename(X_list[-1]).replace('in','bin').replace('jpg','png')
            p = os.path.join(mask_dir, fname)
            if(os.path.exists(p)):
                continue

            Z_list = getFiles(scene_input2_path)
            if (X_list is None):
                raise ValueError('X_list is None')

            # slice frames
            results = checkFrame(X_list)
        
            # load model to segment

            # if large numbers of frames, slice it
            if(results[0]): 
                for rangeee in results[1]: # for each slice
                    slice_X_list =  X_list[rangeee]
                    if category == 'PTZ':
                        slice_Z_list = Z_list[rangeee]
                    else:
                        slice_Z_list = Z_list
                    # load frames for each slice
                    data = generateData(scene_input_path, slice_X_list,slice_Z_list, scene,category)
                
                    # For FgSegNet (multi-scale only) 
                    #Y_proba = model.predict([data[0], data[1], data[2]], batch_size=batch_size, verbose=1) # (xxx, 240, 320, 1)
                
                    # For FgSegNet_v2
                    Y_proba = model.predict(data, batch_size=1, verbose=1)
                    v = Y_proba[fe]
                    #Y_proba = v[0]
                    #Y_proba = model.predict(data, batch_size=1, verbose=1)[0]
                    del data

                    # filter out
                    shape = v.shape
                    #v = v.reshape([shape[0],-1])
                    #if (len(idx)>0): # if have non-ROI
                    #    for i in range(len(v)): # for each frames
                    #        v[i][idx] = 0. # set non-ROI pixel to black
                    #    
                    #v = v.reshape([shape[0], shape[1], shape[2]])

                    prev = 0
                    print ('\n- Saving frames:')
                    for i in range(shape[0]):
                        fname = os.path.basename(slice_X_list[i]).replace('in','bin').replace('jpg','png')
                        x = v[i]
                        y = np.zeros([240,320,3])
                        y[:,:,0] = x[:,:,0]
                        y[:,:,1] = x[:,:,0]
                        y[:,:,2] = x[:,:,0]
                        save_img(os.path.join(mask_dir, fname+'_b.png'), y)
                        
                        x = v[i]
                        y = np.zeros([240,320,3])
                        y[:,:,0] = x[:,:,1]
                        y[:,:,1] = x[:,:,1]
                        y[:,:,2] = x[:,:,1]
                        save_img(os.path.join(mask_dir, fname+'_e.png'), y)
                        
                        
                        #x = Y_proba[1][i]
                        #y = np.zeros([240,320,3])
                        #y[:,:,0] = x[:,:,0]
                        #y[:,:,1] = x[:,:,1]
                        #save_img(os.path.join(mask_dir, fname+'_b.png'), y)
                        
                        #x = Y_proba[1][i]
                        #y = np.zeros([240,320,3])
                        #y[:,:,0] = x[:,:,1]
                        #y[:,:,1] = x[:,:,1]
                        #y[:,:,2] = x[:,:,1]
                        #save_img(os.path.join(mask_dir, fname+'_e.png'), y)

                        sys.stdout.write('\b' * prev)
                        sys.stdout.write('\r')
                        s = str(i+1)
                        sys.stdout.write(s)
                        prev = len(s)
                    
                    del Y_proba, slice_X_list,v

            else: # otherwise, no need to slice
                data = generateData(scene_input_path, X_list,Z_list, scene,category)
            
            
                Y_proba = model.predict(data, batch_size=1, verbose=1)
                v= Y_proba[fe]
                #Y_proba = v[0]
                #Y_proba = model.predict(data, batch_size=1, verbose=1)[0]
            
                del data
                shape = v.shape
                #v = v.reshape([shape[0],-1])
                #if (len(idx)>0): # if have non-ROI
                #        for i in range(len(v)): # for each frames
                #            v[i][idx] = 0. # set non-ROI pixel to black
                #
                #v = v.reshape([shape[0], shape[1], shape[2]])
            
                prev = 0
                print ('\n- Saving frames:')
                for i in range(shape[0]):
                    fname = os.path.basename(X_list[i]).replace('in','bin').replace('jpg','png')
                    x = v[i]
                    y = np.zeros([240,320,3])
                    y[:,:,0] = x[:,:,0]
                    y[:,:,1] = x[:,:,0]
                    y[:,:,2] = x[:,:,0]
                    save_img(os.path.join(mask_dir, fname+'_b.png'), y)
                        
                    x = v[i]
                    y = np.zeros([240,320,3])
                    y[:,:,0] = x[:,:,1]
                    y[:,:,1] = x[:,:,1]
                    y[:,:,2] = x[:,:,1]
                    save_img(os.path.join(mask_dir, fname+'_e.png'), y)
                    
    #                x = Y_proba[1][i]
    #                y = np.zeros([240,320,3])
    #                y[:,:,0] = x[:,:,1]
    #                y[:,:,1] = x[:,:,1]
    #                y[:,:,2] = x[:,:,1]
    #                save_img(os.path.join(mask_dir, fname+'_e.png'), y)
    ##                   
                    sys.stdout.write('\b' * prev)
                    sys.stdout.write('\r')
                    s = str(i+1)
                    sys.stdout.write(s)
                    prev = len(s)
                del Y_proba,v
            del X_list, results

        gc.collect()

mainfunc(4)