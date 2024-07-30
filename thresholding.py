
"""
Created on Mon Jun 27 2018

@author: longang
"""

# coding: utf-8
#get_ipython().magic(u'load_ext autotime')
import numpy as np
import os, glob, sys
from keras.preprocessing import image as kImage
from sklearn.preprocessing import binarize
from keras.preprocessing.image import save_img

## In[]
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


def getFiles(scene_input_path):
    inlist = glob.glob(os.path.join(scene_input_path,'*.png'))
    return np.asarray(inlist)

def generateData(X_list):
    # read images
    X = []
    prev = 0
    print ('\n\t- Loading probability masks:')
    for i in range(0, len(X_list)):
        x = kImage.load_img(X_list[i], grayscale=True)
        x = kImage.img_to_array(x)
        x /= 255.
        X.append(x)
        sys.stdout.write('\b' * prev)
        sys.stdout.write('\r')
        s = str(i+1)
        sys.stdout.write(s)
        prev = len(s)
    
    del x, X_list
    X = np.asarray(X)
    return X


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

#try:
ths = [0.3,0.4,0.5,0.6,0.7,0.8,0.9] # threshold to apply [0.4, 0.5, 0.6, 0.7, 0.8, 0.9] or anyting btw
num_frames = 200 #threshold for 25frames

results_dir = os.path.join('ResSegNet', 'resultsALLCL200')
#results_dir = 'C:\\Users\\islam\\source\\repos\\MOD2\\MOD2\\ResSegNet\\results2_cdnetboth200'
#th_results_dir = os.path.join('ResSegNet', 'results2_cdnetboth200' + str(num_frames) + '_th' + str(th))

for q in range(len(ths)):

    th_results_dir = os.path.join('ResSegNet', 'resultsALLCL200' + str(ths[q]))
    
    for category, scene_list in dataset.items():
        for scene in scene_list:
            #if scene != 'traffic':
            #    continue
            th = ths[q]
            print ('\n->>> ' + category + ' / ' + scene)
            save_th_mask_to_dir = os.path.join(th_results_dir, category, scene)
            if not os.path.exists(save_th_mask_to_dir):
                os.makedirs(save_th_mask_to_dir)
                
            scene_mask_path = os.path.join(results_dir, category, scene)
            X_list = getFiles(scene_mask_path)
            if (X_list is None):
                raise ValueError('X_list is None')
            results = checkFrame(X_list)
            
            if(results[0]): # if large numbers of frames, slice it
                for rangeee in results[1]: # for each slice
                    slice_X_list =  X_list[rangeee]
                    mask_slice = generateData(slice_X_list)
                    shape = mask_slice.shape # (1000,240,320,1)
                    mask_slice = mask_slice.reshape([shape[0],-1])
                    mask_slice = binarize(mask_slice, threshold = th)
                    mask_slice = mask_slice.reshape([shape[0], shape[1], shape[2]])
    
                    prev = 0
                    print ('\n\t- Saving thresholding masks:')
                    for i in range(shape[0]):
                        fname = os.path.basename(slice_X_list[i])
                        y = np.zeros([shape[1],shape[2],3])
                        
                        y[:,:,0] = mask_slice[i]
                        y[:,:,1] = mask_slice[i]
                        y[:,:,2] = mask_slice[i]
                        save_img(os.path.join(save_th_mask_to_dir, fname), y)
                        sys.stdout.write('\b' * prev)
                        sys.stdout.write('\r')
                        s = str(i+1)
                        sys.stdout.write(s)
                        prev = len(s)
                    del mask_slice, slice_X_list
            else: # otherwise, no need to slice
                mask = generateData(X_list)
                shape = mask.shape
                mask = mask.reshape([shape[0],-1])
                mask = binarize(mask, threshold = th)
                mask = mask.reshape([shape[0], shape[1], shape[2]])
                
                prev = 0
                print ('\n\t- Saving thresholding masks:')
                for i in range(shape[0]):
                    fname = os.path.basename(X_list[i])
                    y = np.zeros([shape[1],shape[2],3])
                    
                    y[:,:,0] = mask[i]
                    y[:,:,1] = mask[i]
                    y[:,:,2] = mask[i]
                    save_img(os.path.join(save_th_mask_to_dir, fname), y)
                    sys.stdout.write('\b' * prev)
                    sys.stdout.write('\r')
                    s = str(i+1)
                    sys.stdout.write(s)
                    prev = len(s)
                del mask
            del X_list, results