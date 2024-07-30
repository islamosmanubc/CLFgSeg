

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
from Resnet_CD import ResSegModel
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

sequences = ['highway', 
                   'pedestrians',
                   'office',
                   'PETS2006','badminton',
                   'traffic',
                   'boulevard', 
                   'sidewalk','skating', 
                   'blizzard',
                   'snowFall',
                   'wetSnow','boats',
                   'canoe',
                   'fall',
                   'fountain01',
                   'fountain02',
                    'overpass','abandonedBox',
                    'parking',
                    'sofa',
                    'streetLight',
                    'tramstop',
                    'winterDriveway','port_0_17fps',
                    'tramCrossroad_1fps',
                    'tunnelExit_0_35fps',
                    'turnpike_0_5fps','bridgeEntry',
                    'busyBoulvard',
                    'fluidHighway',
                    'streetCornerAtNight',
                    'tramStation',
                    'winterStreet',
                    'continuousPan',
                   'intermittentPan',
                   'twoPositionPTZCam',
                   'zoomInZoomOut',
                    'backdoor',
                    'bungalows',
                    'busStation',
                    'copyMachine',
                    'cubicle',
                    'peopleInShade','corridor',
                   'diningRoom',
                   'lakeSide',
                   'library',
                   'park','turbulence0',
                    'turbulence1',
                   'turbulence2',
                   'turbulence3']

# Extract all mask
dataset = {
           'baseline':[
                   'highway', 
                   'pedestrians',
                   'office',
                   'PETS2006'
                   ],
           'cameraJitter':[
                   'badminton',
                   'traffic',
                   'boulevard', 
                   'sidewalk'
                   ],
           'badWeather':[
                   'skating', 
                   'blizzard',
                   'snowFall',
                   'wetSnow'
                   ],
            'dynamicBackground':[
                   'boats',
                   'canoe',
                   'fall',
                   'fountain01',
                   'fountain02',
                    'overpass'
                    ],
            'intermittentObjectMotion':[
                    'abandonedBox',
                    'parking',
                    'sofa',
                    'streetLight',
                    'tramstop',
                    'winterDriveway'
                    ],
            'lowFramerate':[
                    'port_0_17fps',
                    'tramCrossroad_1fps',
                    'tunnelExit_0_35fps',
                    'turnpike_0_5fps'
                    ],
            'nightVideos':[
                    'bridgeEntry',
                    'busyBoulvard',
                    'fluidHighway',
                    'streetCornerAtNight',
                    'tramStation',
                    'winterStreet'
                    ],
           'PTZ':[
                   'continuousPan',
                   'intermittentPan',
                   'twoPositionPTZCam',
                   'zoomInZoomOut'
                   ],
            'shadow':[
                    'backdoor',
                    'bungalows',
                    'busStation',
                    'copyMachine',
                    'cubicle',
                    'peopleInShade'
                    ],
           'thermal':[
                   'corridor',
                   'diningRoom',
                   'lakeSide',
                   'library',
                   'park'
                   ],
            'turbulence':[
                    'turbulence0',
                    'turbulence1',
                   'turbulence2',
                   'turbulence3'
                    ] 
}

# number of exp frame (25, 50, 200)
num_frames = 200

# 1. Raw RGB frame to extract foreground masks, downloaded from changedetection.net
raw_dataset_dir = 'D:/phd/moving_object_detection/code/FgSegNet_v2_master/datasets/CDnet2014_dataset'

# 2. model dir

# 3. path to store results
#results_dir = os.path.join('ResSegNet', 'results4CL' + str(num_frames))
results_dir = os.path.join('ResSegNet', 'weights_dense_pretrain_' + str(num_frames))

ROI_file = 'D:/phd/moving_object_detection/code/FgSegNet_v2_master/datasets/CDnet2014_dataset/badWeather/blizzard/ROI.jpg'
img = kImage.load_img(ROI_file, target_size=[240,320])
img = kImage.img_to_array(img)
img_shape = img.shape
model = ResSegModel(0, (240,320,6), 'x')
model = model.initModel('CDnet')


# Loop through all categories (e.g. baseline)
c=0
for category, scene_list in dataset.items():
    
    if category == 'dynamicBackground':
        c = c+1
    if category == 'nightVideos':
        c = c+1
    if category == 'thermal':
        c = c+1

    #if category == 'badWeather':
    #    c = 0
    #if category == 'baseline':
    #    c = 0
    #if category == 'cameraJitter':
    #    c = 0
    #if category == 'dynamicBackground':
    #    c = 0
    #if category == 'intermittentObjectMotion':
    #    c = 0
    #if category == 'lowFramerate':
    #    c = 0
    #if category == 'nightVideos':
    #    c = 1
    #if category == 'PTZ':
    #    c = 1
    #if category == 'shadow':
    #    c = 1
    #if category == 'thermal':
    #    c = 1
    #if category == 'turbulence':
    #    c = 1
    weights = np.load('weights_dense_pretrain_200.npy', allow_pickle=True, encoding="latin1").tolist()
    #weights = np.load('weights_dense_gen200_'+str(c)+'.npy', allow_pickle=True, encoding="latin1").tolist()
    a = np.array(model.get_weights())
    for k in range(len(weights)):
        a[k] = weights[k]
    
    model.set_weights(a)
    #b = np.array(model.get_weights())

    # Loop through all scenes (e.g. highway, ...)
    for scene in scene_list:
        print ('\n->>> ' + category + ' / ' + scene)
        

        #if scene == 'blizzard':
        #    c = 0
        #if scene == 'skating':
        #    c = 0
        #if scene == 'wetSnow':
        #    c = 1
        #if scene == 'snowFall':
        #    c = 1
        #if scene == 'highway':
        #    c = 2
        #if scene == 'office':
        #    c = 2
        #if scene == 'pedestrians':
        #    c = 3
        #if scene == 'PETS2006':
        #    c = 3
        #if scene == 'badminton':
        #    c = 4
        #if scene == 'boulevard':
        #    c = 4
        #if scene == 'sidewalk':
        #    c = 5
        #if scene == 'traffic':
        #    c = 5
        #if scene == 'boats':
        #    c = 6
        #if scene == 'canoe':
        #    c = 6
        #if scene == 'fall':
        #    c = 7
        #if scene == 'fountain01':
        #    c = 7
        #if scene == 'fountain02':
        #    c = 8
        #if scene == 'overpass':
        #    c = 8
        #if scene == 'abandonedBox':
        #    c = 9
        #if scene == 'parking':
        #    c = 9
        #if scene == 'sofa':
        #    c = 10
        #if scene == 'streetLight':
        #    c = 10
        #if scene == 'tramstop':
        #    c = 11
        #if scene == 'winterDriveway':
        #    c = 11
        #if category == 'lowFramerate':
        #    c = 12
        #if scene == 'bridgeEntry':
        #    c = 13
        #if scene == 'busyBoulvard':
        #    c = 13
        #if scene == 'streetCornerAtNight':
        #    c = 14
        #if scene == 'tramStation':
        #    c = 14
        #if scene == 'winterStreet':
        #    c = 14
        #if scene == 'fluidHighway':
        #    c = 14
        #if category == 'PTZ':
        #    c = 15
        #if category == 'shadow':
        #    c = 16
        #if category == 'turbulence':
        #    c = 19
        #if scene == 'library':
        #    c = 18
        #if scene == 'park':
        #    c = 18
        #if scene == 'corridor':
        #    c = 17
        #if scene == 'diningRoom':
        #    c = 17
        #if scene == 'lakeSide':
        #    c = 17
        #weights = np.load('weights_cl20_dense_gen25_'+str(c)+'.npy', allow_pickle=True, encoding="latin1").tolist()
        #a = np.array(model.get_weights())
        #for k in range(len(weights)):
        #    a[k] = weights[k]
        #model.set_weights(a)


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
                v = model.predict(data, batch_size=1, verbose=1)
                Y_proba = v
                #Y_proba = model.predict(data, batch_size=1, verbose=1)[0]
                del data

                # filter out
                shape = Y_proba.shape
                Y_proba = Y_proba.reshape([shape[0],-1])
                if (len(idx)>0): # if have non-ROI
                    for i in range(len(Y_proba)): # for each frames
                        Y_proba[i][idx] = 0. # set non-ROI pixel to black
                        
                Y_proba = Y_proba.reshape([shape[0], shape[1], shape[2]])

                prev = 0
                print ('\n- Saving frames:')
                for i in range(shape[0]):
                    fname = os.path.basename(slice_X_list[i]).replace('in','bin').replace('jpg','png')
                    x = Y_proba[i]
                    y = np.zeros([240,320,3])
                    y[:,:,0] = x
                    y[:,:,1] = x
                    y[:,:,2] = x
                    save_img(os.path.join(mask_dir, fname), y)

                    sys.stdout.write('\b' * prev)
                    sys.stdout.write('\r')
                    s = str(i+1)
                    sys.stdout.write(s)
                    prev = len(s)
                    
                del Y_proba, slice_X_list

        else: # otherwise, no need to slice
            data = generateData(scene_input_path, X_list,Z_list, scene,category)
            
            
            v = model.predict(data, batch_size=1, verbose=1)
            Y_proba = v
            #Y_proba = model.predict(data, batch_size=1, verbose=1)[0]
            
            del data
            shape = Y_proba.shape
            Y_proba = Y_proba.reshape([shape[0],-1])
            if (len(idx)>0): # if have non-ROI
                    for i in range(len(Y_proba)): # for each frames
                        Y_proba[i][idx] = 0. # set non-ROI pixel to black

            Y_proba = Y_proba.reshape([shape[0], shape[1], shape[2]])
            
            prev = 0
            print ('\n- Saving frames:')
            for i in range(shape[0]):
                fname = os.path.basename(X_list[i]).replace('in','bin').replace('jpg','png')
                x = Y_proba[i]
                y = np.zeros([240,320,3])
                y[:,:,0] = x
                y[:,:,1] = x
                y[:,:,2] = x
#                if batch_size in [2,4] and scene=='badminton':
#                    x = imresize(x, (480,720), interp='nearest')
#                
#                if batch_size in [2,4] and scene=='PETS2006':
#                        x = imresize(x, (576,720), interp='nearest')
                        
                save_img(os.path.join(mask_dir, fname), y)
                sys.stdout.write('\b' * prev)
                sys.stdout.write('\r')
                s = str(i+1)
                sys.stdout.write(s)
                prev = len(s)
            del Y_proba
        del X_list, results

    gc.collect()