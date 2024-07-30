

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
#from LongLifeLearningTree_res import LLLTree
from Resnet_CDMLGen import ResSegModel
#from skimage.transform import pyramid_gaussian
from keras.models import load_model
#from scipy.misc import imsave#, imresize
import gc
from PIL import Image


from collections import OrderedDict

def mse(actual, pred): 
    actual, pred = np.array(actual), np.array(pred)
    return np.square(np.subtract(actual,pred)).mean() 
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

def getFiles(scene_input_path,png=False):
    inlist = glob.glob(os.path.join(scene_input_path,'*.jpg'))
    if png:
        inlist = glob.glob(os.path.join(scene_input_path,'*.png'))
    return np.asarray(inlist)


def TestDataset(sequences,initial_dataset,target_dataset,raw_dataset_dir,samples_path,samples_gt_path,png,kshots,itr=5):
    
    results_dir = os.path.join('D:/paper7res', 'refnet_'+initial_dataset+'_'+target_dataset+'_'+str(itr)+'_'+str(kshots))
    
    model = ResSegModel(1e-4, (240,320,6), 'x')
    model = model.initModel('CDnet')
    
    # Loop through all scenes (e.g. highway, ...)
    for scene in sequences:
        print ('\n->>> '+scene)
    
        

        Y_list = []
        X_list = []
        Z_list = []
        E_list = []
    

        mask_dir = os.path.join(results_dir, scene)
        if not os.path.exists(mask_dir):
            os.makedirs(mask_dir)

        
        
        
        
        Y_list.append(glob.glob(os.path.join(samples_gt_path, scene, '*.png')))
        
        if target_dataset == 'davis17':
            X_list.append(glob.glob(os.path.join(samples_path, scene,'*.jpg')))
            #Z_list.append(glob.glob(os.path.join(samples_path, scene, '*.jpg')))
            E_list.append(glob.glob(os.path.join('datasets/davis17/DAVIStrain/Annotations/edge',scene,'*.png')))
        elif not png:
            X_list.append(glob.glob(os.path.join(samples_path, scene, 'input','*.jpg')))
            #Z_list.append(glob.glob(os.path.join(samples_path, scene, 'back','*.jpg')))
            E_list.append(glob.glob(os.path.join(samples_gt_path, scene, 'edge','*.png')))
        else:
            X_list.append(glob.glob(os.path.join(samples_path, scene, 'input','*.png')))
            #Z_list.append(glob.glob(os.path.join(samples_path, scene, 'back','*.png')))
            E_list.append(glob.glob(os.path.join(samples_gt_path, scene, 'edge','*.png')))

            
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

        x = X_list[0]
        x = np.asarray(x)
    
        if len(x) < kshots:
            kshots = 1
        k = np.random.randint(0,len(X_list[0]),kshots)
        x = x[k]
        X = []
        Z = []
        
        img = kImage.load_img(x[0],target_size = [240,320,3])
        img = kImage.img_to_array(img)

        for i in range(len(x)):
            im = kImage.load_img(x[i],target_size = [240,320,3])
            im = kImage.img_to_array(im)
            X.append(im)
            Z.append(img)

        x = np.asarray(X)
        z = np.asarray(Z)
        
        
    
        void_label = -1. # non-ROI
        y = np.asarray(Y_list[0])
        y = y[k]
        Y = []
        for i in range(len(y)):
            im = kImage.load_img(y[i], grayscale = True,target_size = [240,320,1])
            im = kImage.img_to_array(im)
            shape = im.shape
            im /= 255.0
            im = im.reshape(-1)
            idx = np.where(np.logical_and(im>0.25, im<0.8))[0] # find non-ROI
            if (len(idx)>0):
                im[idx] = void_label
            im = im.reshape(shape)
            im = np.floor(im)
            Y.append(im)
    
        y = np.asarray(Y)

        e = np.asarray(E_list[0])
        e = e[k]
        E = []
        for i in range(len(e)):
            im = kImage.load_img(e[i], grayscale = True,target_size = [240,320,1])
            im = kImage.img_to_array(im)
            shape = im.shape
            im /= 255.0
            im = im.reshape(-1)
            idx = np.where(np.logical_and(im>0.25, im<0.8))[0] # find non-ROI
            if (len(idx)>0):
                im[idx] = void_label
            im = im.reshape(shape)
            im = np.floor(im)
            E.append(im)

        e = np.asarray(E)
        
        X = np.append(x,z,axis=3)
        Y = np.append(y,e,axis=3)

        if target_dataset == 'davis17':
            scene_input_path = os.path.join(raw_dataset_dir, scene)
            scene_input2_path = os.path.join(raw_dataset_dir, scene)
        else:
            scene_input_path = os.path.join(raw_dataset_dir, scene, 'input')
            scene_input2_path = os.path.join(raw_dataset_dir, scene, 'back')
    
    
        # load path of files
        X_list = getFiles(scene_input_path,png)
        fname = os.path.basename(X_list[-1]).replace('jpg','png')
        p = os.path.join(mask_dir, fname)
        if(os.path.exists(p)):
            continue
        Z_list = getFiles(scene_input2_path,png)
        ztemp = []
        ztemp.append(Z_list[0])
        Z_list = ztemp

        if (X_list is None):
            raise ValueError('X_list is None')

        # slice frames
        results = checkFrame(X_list)
        
        weights = np.load('weights_'+initial_dataset+'.npy', allow_pickle=True, encoding="latin1").tolist()
        #weights = np.load('weights_segtrack1_cl2'+str(tasknum)+'_'+str(t)+'.npy', allow_pickle=True, encoding="latin1").tolist()
        a = np.array(model.get_weights())
        for k in range(len(weights)):
            a[k] = weights[k]
        model.set_weights(a)


        
        model.fit(X, [Y[:,:,:,:1],Y,Y,Y,Y], 
                      epochs=itr, batch_size=1, 
                      verbose=1, shuffle = True)
        

        # if large numbers of frames, slice it
        if(results[0]): 
            for rangeee in results[1]: # for each slice
                slice_X_list =  X_list[rangeee]
                slice_Z_list = Z_list
                # load frames for each slice
                data = generateData(scene_input_path, slice_X_list,slice_Z_list, scene,'')
                
                # For FgSegNet (multi-scale only) 
                #Y_proba = model.predict([data[0], data[1], data[2]], batch_size=batch_size, verbose=1) # (xxx, 240, 320, 1)
                
                # For FgSegNet_v2
                v = model.predict(data, batch_size=1, verbose=1)
                Y_proba = v[0]
                #Y_proba = model.predict(data, batch_size=1, verbose=1)[0]
                del data

                # filter out
                shape = Y_proba.shape
                Y_proba = Y_proba.reshape([shape[0],-1])
                    
                Y_proba = Y_proba.reshape([shape[0], shape[1], shape[2]])

                prev = 0
                print ('\n- Saving frames:')
                for i in range(shape[0]):
                    fname = os.path.basename(slice_X_list[i])
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
            data = generateData(scene_input_path, X_list,Z_list, scene,'')
            
            v = model.predict(data, batch_size=1, verbose=1)
            Y_proba = v[0]
            #Y_proba = model.predict(data, batch_size=1, verbose=1)[0]
            
            del data
            shape = Y_proba.shape
            Y_proba = Y_proba.reshape([shape[0],-1])

            Y_proba = Y_proba.reshape([shape[0], shape[1], shape[2]])
            
            prev = 0
            print ('\n- Saving frames:')
            for i in range(shape[0]):
                fname = os.path.basename(X_list[i])
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
            del Y_proba
        del X_list, results
    gc.collect()

def Test(kshots):

    sequences_davis = ['bear','blackswan','bmx-bumps','bmx-trees','boat','breakdance','breakdance-flare','bus','camel',
                   'car-roundabout','car-shadow','car-turn','cows','dance-jump','dance-twirl','dog','dog-agility',
                   'drift-chicane','drift-straight','drift-turn','elephant','flamingo','goat','hike','hockey','horsejump-high',
                   'horsejump-low','kite-surf','kite-walk','libby','lucia','mallard-fly','mallard-water','motocross-bumps',
                   'motocross-jump','motorbike','paragliding','paragliding-launch','parkour','rhino','rollerblade','scooter-black',
                   'scooter-gray','soapbox','soccerball','stroller','surf','swing','tennis','train']

    sequences_seg = ['bird_of_paradise','birdfall','bmx','cheetah','drift','frog','girl','hummingbird','monkey',
               'monkeydog','parachute','penguin','soldier','worm']

    sequences_davis17 = ['bear','blackswan','bike-packing','bmx-bumps','bmx-trees','boat','boxing-fisheye','breakdance'
           ,'breakdance-flare','bus','camel','car-roundabout','car-shadow','car-turn','cat-girl','classic-car'
           ,'color-run','cows','crossing','dance-jump','dance-twirl','dancing','disc-jockey','dog','dog-agility','dog-gooses','dogs-jump',
           'dogs-scale','drift-chicane','drift-straight','drift-turn','drone','elephant','flamingo','goat',
           'gold-fish','hike','hockey','horsejump-high','horsejump-low','india','judo','kid-football','kite-surf',
           'kite-walk','koala','lab-coat','lady-running','libby','lindy-hop','loading','longboard','lucia','mallard-fly',
           'mallard-water','mbike-trick','miami-surf','motocross-bumps','motocross-jump','motorbike','night-race',
           'paragliding','paragliding-launch','parkour','pigs','planes-water','rallye','rhino','rollerblade','schoolgirls',
           'scooter-black','scooter-board','scooter-gray','sheep','shooting','skate-park','snowboard','soapbox',
           'soccerball','stroller','stunt','surf','swing','tennis','tractor-sand','train','tuk-tuk','upside-down',
           'varanus-cage','walking']

    # 1. Raw RGB frame to extract foreground masks, downloaded from changedetection.net
    raw_dataset_dir_davis = 'datasets/DAVIS/480pfull'
    raw_dataset_dir_seg = 'datasets/segtrackv2/full'
    raw_dataset_dir_davis17 = 'datasets/davis17/DAVIStrain1/JPEGImages/input'

    # 2. model dir

    # 3. path to store results
    #results_dir = os.path.join('ResSegNet', 'results4CL' + str(num_frames))

    
    samples_gt_path_davis = 'datasets/DAVIS/480pY'
    samples_path_davis = 'datasets/DAVIS/480p'
    
    samples_gt_path_seg = 'datasets/segtrackv2/GroundTruth'
    samples_path_seg = 'datasets/segtrackv2/JPEGImages1'

    samples_gt_path_davis17 = 'datasets/davis17/DAVIStrain/Annotations/GT'
    samples_path_davis17 = 'datasets/davis17/DAVIStrain/JPEGImages/input'

    itrs = [10,20,50,100]
    itrs = [5]
    for i in range(len(itrs)):
        itr = itrs[i]
        TestDataset(sequences_davis,'davis','davis',raw_dataset_dir_davis,samples_path_davis,samples_gt_path_davis,False,kshots,itr)
        TestDataset(sequences_seg,'davis','segtrack',raw_dataset_dir_seg,samples_path_seg,samples_gt_path_seg,True,kshots,itr)
        TestDataset(sequences_davis17,'davis','davis17',raw_dataset_dir_davis17,samples_path_davis17,samples_gt_path_davis17,False,kshots,itr)
        
        TestDataset(sequences_davis,'davis17','davis',raw_dataset_dir_davis,samples_path_davis,samples_gt_path_davis,False,kshots,itr)
        TestDataset(sequences_seg,'davis17','segtrack',raw_dataset_dir_seg,samples_path_seg,samples_gt_path_seg,True,kshots,itr)
        TestDataset(sequences_davis17,'davis17','davis17',raw_dataset_dir_davis17,samples_path_davis17,samples_gt_path_davis17,False,kshots,itr)

        TestDataset(sequences_davis,'segtrack','davis',raw_dataset_dir_davis,samples_path_davis,samples_gt_path_davis,False,kshots,itr)
        TestDataset(sequences_seg,'segtrack','segtrack',raw_dataset_dir_seg,samples_path_seg,samples_gt_path_seg,True,kshots,itr)
        TestDataset(sequences_davis17,'segtrack','davis17',raw_dataset_dir_davis17,samples_path_davis17,samples_gt_path_davis17,False,kshots,itr)


Test(1)
Test(5)
Test(10)