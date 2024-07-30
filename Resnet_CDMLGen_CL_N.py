
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 2018

@author: longang
"""

import keras
from keras.models import Model
from keras.layers import Input, Dropout, Activation, SpatialDropout2D,Concatenate
from keras.layers.convolutional import Conv2D, Cropping2D, UpSampling2D
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.layers import concatenate, add, multiply
from my_upsampling_2d import MyUpSampling2D
from instance_normalization import InstanceNormalization
import keras.backend as K
import tensorflow as tf

def loss(y_true, y_pred):
    void_label = -1.
    y_pred = K.reshape(y_pred, [-1])
    y_true = K.reshape(y_true, [-1])
    idx = tf.where(tf.not_equal(y_true, tf.constant(void_label, dtype=tf.float32)))
    y_pred = tf.gather_nd(y_pred, idx) 
    y_true = tf.gather_nd(y_true, idx)
    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

def acc(y_true, y_pred):
    void_label = -1.
    y_pred = tf.reshape(y_pred, [-1])
    y_true = tf.reshape(y_true, [-1])
    idx = tf.where(tf.not_equal(y_true, tf.constant(void_label, dtype=tf.float32)))
    y_pred = tf.gather_nd(y_pred, idx) 
    y_true = tf.gather_nd(y_true, idx)
    return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)

def loss2(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

def acc2(y_true, y_pred):
    return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)

class ResSegModel(object):
    
    def __init__(self, lr, img_shape, scene,tasks):
        self.lr = lr
        self.img_shape = img_shape
        self.scene = scene
        self.method_name = 'FgSegNet_v2'
        self.Tasks = tasks
        
    def Resnet(self, inp): 
        
        # Block 1
        Tasks = self.Tasks
        tp = int(128/Tasks)
        ntp = 128-tp
        nt = Conv2D(ntp, (3, 3), activation='relu', padding='same', name='block1_conv1NT', data_format='channels_last',trainable = False)(inp)
        t = Conv2D(tp, (3, 3), activation='relu', padding='same', name='block1_conv1T', data_format='channels_last')(inp)
        x = Concatenate(axis=3)([nt,t])
        nt = Conv2D(ntp, (3, 3), activation='relu', padding='same', name='block1_conv2NT',trainable = False)(x)
        t = Conv2D(tp, (3, 3), activation='relu', padding='same', name='block1_conv2T')(x)
        x = Concatenate(axis=3)([nt,t])
        
        nt = Conv2D(ntp, (3, 3), activation='relu', padding='same', name='block1_convresNT', data_format='channels_last',trainable = False)(inp)
        t = Conv2D(tp, (3, 3), activation='relu', padding='same', name='block1_convresT', data_format='channels_last')(inp)
        xr = Concatenate(axis=3)([nt,t])
        x = add([x, xr])
        nt = Conv2D(ntp, (3, 3), activation='relu', padding='same', name='block1_conv3NT',trainable = False)(x)
        t = Conv2D(tp, (3, 3), activation='relu', padding='same', name='block1_conv3T')(x)
        x = Concatenate(axis=3)([nt,t])
        a = x
        x1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    
        # Block 2
        s = 128
        tp = int(s/Tasks)
        ntp = s-tp
        nt = Conv2D(ntp, (3, 3), activation='relu', padding='same', name='block2_conv1NT',trainable = False)(x1)
        t = Conv2D(tp, (3, 3), activation='relu', padding='same', name='block2_conv1T')(x1)
        x = Concatenate(axis=3)([nt,t])
        
        nt = Conv2D(ntp, (3, 3), activation='relu', padding='same', name='block2_conv2NT',trainable = False)(x)
        t = Conv2D(tp, (3, 3), activation='relu', padding='same', name='block2_conv2T')(x)
        x = Concatenate(axis=3)([nt,t])
        
        nt = Conv2D(ntp, (3, 3), activation='relu', padding='same', name='block2_convresNT',trainable = False)(x1)
        t = Conv2D(tp, (3, 3), activation='relu', padding='same', name='block2_convresT')(x1)
        xr = Concatenate(axis=3)([nt,t])
        x = add([x, xr])
        nt = Conv2D(ntp, (3, 3), activation='relu', padding='same', name='block2_conv3NT',trainable = False)(x)
        t = Conv2D(tp, (3, 3), activation='relu', padding='same', name='block2_conv3T')(x)
        x = Concatenate(axis=3)([nt,t])
        b = x
        x2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    
        # Block 3
        s = 128
        tp = int(s/Tasks)
        ntp = s-tp
        nt = Conv2D(ntp, (3, 3), activation='relu', padding='same', name='block3_conv1NT',trainable = False)(x2)
        t = Conv2D(tp, (3, 3), activation='relu', padding='same', name='block3_conv1T')(x2)
        x = Concatenate(axis=3)([nt,t])
        
        nt = Conv2D(ntp, (3, 3), activation='relu', padding='same', name='block3_conv2NT',trainable = False)(x)
        t = Conv2D(tp, (3, 3), activation='relu', padding='same', name='block3_conv2T')(x)
        x = Concatenate(axis=3)([nt,t])
        
        nt = Conv2D(ntp, (3, 3), activation='relu', padding='same', name='block3_convresNT',trainable = False)(x2)
        t = Conv2D(tp, (3, 3), activation='relu', padding='same', name='block3_convresT')(x2)
        xr = Concatenate(axis=3)([nt,t])
        
        x = add([x, xr])
        nt = Conv2D(ntp, (3, 3), activation='relu', padding='same', name='block3_conv3NT',trainable = False)(x)
        t = Conv2D(tp, (3, 3), activation='relu', padding='same', name='block3_conv3T')(x)
        x = Concatenate(axis=3)([nt,t])
        
        x3 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    
        # Block 4
        
        s = 128
        tp = int(s/Tasks)
        ntp = s-tp
        nt = Conv2D(ntp, (3, 3), activation='relu', padding='same', name='block4_conv1NT',trainable = False)(x3)
        t = Conv2D(tp, (3, 3), activation='relu', padding='same', name='block4_conv1T')(x3)
        x = Concatenate(axis=3)([nt,t])

        x = Dropout(0.5, name='dr1')(x)
        nt = Conv2D(ntp, (3, 3), activation='relu', padding='same', name='block4_conv2NT',trainable = False)(x)
        t = Conv2D(tp, (3, 3), activation='relu', padding='same', name='block4_conv2T')(x)
        x = Concatenate(axis=3)([nt,t])

        nt = Conv2D(ntp, (3, 3), activation='relu', padding='same', name='block4_convresNT',trainable = False)(x3)
        t = Conv2D(tp, (3, 3), activation='relu', padding='same', name='block4_convresT')(x3)
        xr = Concatenate(axis=3)([nt,t])

        x = add([x, xr])
        nt = Conv2D(ntp, (3, 3), activation='relu', padding='same', name='block4_conv3NT',trainable = False)(x)
        t = Conv2D(tp, (3, 3), activation='relu', padding='same', name='block4_conv3T')(x)
        x = Concatenate(axis=3)([nt,t])

        x = Dropout(0.5, name='dr2')(x)

        return x, a, b
    
    def decoder(self,x,a,b):
        Tasks = self.Tasks
        s = 128
        tp = int(s/Tasks)
        ntp = s-tp
        a = GlobalAveragePooling2D()(a)
        nt = Conv2D(ntp, (1, 1), strides=1, padding='same',trainable = False)(b)
        t = Conv2D(tp, (1, 1), strides=1, padding='same')(b)
        b = Concatenate(axis=3)([nt,t])
        b = GlobalAveragePooling2D()(b)
        
        #block1
        nt = Conv2D(ntp, (3, 3), strides=1, padding='same',trainable = False)(x)
        t = Conv2D(tp, (3, 3), strides=1, padding='same')(x)
        x = Concatenate(axis=3)([nt,t])
        x = InstanceNormalization()(x)
        x = Activation('relu')(x)
        x1 = multiply([x, b])
        x = add([x, x1])
        x = UpSampling2D(size=(2, 2))(x)
        
        o1 = UpSampling2D(size=(2, 2))(x)
        o1 = UpSampling2D(size=(2, 2))(o1)
        o1 = Conv2D(1, (3,3), strides=1, padding='same', activation='sigmoid', name = 'block1out',trainable = False)(o1)
        
        #block2
        s = 128
        tp = int(s/Tasks)
        ntp = s-tp
        nt = Conv2D(ntp, (3, 3), strides=1, padding='same',trainable = False)(x)
        t = Conv2D(tp, (3, 3), strides=1, padding='same')(x)
        x = Concatenate(axis=3)([nt,t])
        x = InstanceNormalization()(x)
        x = Activation('relu')(x)
        x2 = multiply([x, a])
        x = add([x, x2])
        xs2 = UpSampling2D(size=(2, 2))(x)
        
        o2 = UpSampling2D(size=(2, 2))(xs2)
        o2 = Conv2D(1, (3,3), strides=1, padding='same', activation='sigmoid', name = 'block2out',trainable=False)(o2)
        
        #block3T1
        s = 128
        tp = int(s/Tasks)
        ntp = s-tp
        nt = Conv2D(ntp, (3, 3), strides=1, padding='same',trainable = False)(xs2)
        t = Conv2D(tp, (3, 3), strides=1, padding='same')(xs2)
        x = Concatenate(axis=3)([nt,t])
        x = InstanceNormalization()(x)
        x = Activation('relu')(x)
        x = UpSampling2D(size=(2, 2))(x)
        #block4T1
        o3 = Conv2D(1, (3,3), strides=1, padding='same', activation='sigmoid', name = 'block3out',trainable=False)(x)
        

        s = 128
        tp = int(s/Tasks)
        ntp = s-tp
        c = Concatenate(axis=3)([x,o1,o2,o3])
        nt = Conv2D(ntp, (3, 3), activation='relu', padding='same', name='blockD_conv1NT',trainable=False)(c)
        t = Conv2D(tp, (3, 3), activation='relu', padding='same', name='blockD_conv1T')(c)
        x = Concatenate(axis=3)([nt,t])
        
        nt = Conv2D(ntp, (3, 3), activation='relu', padding='same', name='blockD_conv2NT',trainable=False)(x)
        t = Conv2D(tp, (3, 3), activation='relu', padding='same', name='blockD_conv2T')(x)
        x = Concatenate(axis=3)([nt,t])
        nt = Conv2D(ntp, (3, 3), activation='relu', padding='same', name='blockD_convresNT',trainable=False)(c)
        t = Conv2D(tp, (3, 3), activation='relu', padding='same', name='blockD_convresT')(c)
        xr = Concatenate(axis=3)([nt,t])
        x = add([x, xr])
        nt = Conv2D(ntp, (3, 3), activation='relu', padding='same', name='blockD_conv3NT',trainable=False)(x)
        t = Conv2D(tp, (3, 3), activation='relu', padding='same', name='blockD_conv3T')(x)
        x = Concatenate(axis=3)([nt,t])
        
        o4 = Conv2D(1, (3,3), strides=1, padding='same', activation='sigmoid', name = 'block4out',trainable=False)(x)

        s = 128
        tp = int(s/Tasks)
        ntp = s-tp
        c = Concatenate(axis=3)([x,o4])
        nt = Conv2D(ntp, (3, 3), activation='relu', padding='same', name='blockM_conv1NT',trainable =False)(c)
        t = Conv2D(tp, (3, 3), activation='relu', padding='same', name='blockM_conv1T')(c)
        x = Concatenate(axis=3)([nt,t])
        o5 = Conv2D(1, (3,3), strides=1, padding='same', activation='sigmoid', name = 'merged',trainable=False)(x)
        return o5,o4,o3,o2,o1
    

    def M_FPM(self, x):
        
        Tasks = self.Tasks
        s = 128
        tp = int(s/Tasks)
        ntp = s-tp
        pool = MaxPooling2D((2, 2), strides=(1,1), padding='same')(x)
        nt = Conv2D(ntp, (1, 1), padding='same',trainable = False)(pool)
        t = Conv2D(tp, (1, 1), padding='same')(pool)
        pool = Concatenate(axis=3)([nt,t])

        nt = Conv2D(ntp, (3, 3), padding='same',trainable = False)(x)
        t = Conv2D(tp, (3, 3), padding='same')(x)
        d1 = Concatenate(axis=3)([nt,t])
        
        y = concatenate([x,pool, d1], axis=-1, name='cat4')
        y = Activation('relu')(y)
        nt = Conv2D(ntp, (3, 3), padding='same', dilation_rate=4,trainable = False)(y)
        t = Conv2D(tp, (3, 3), padding='same', dilation_rate=4)(y)
        d4 = Concatenate(axis=3)([nt,t])
        
        y = concatenate([x,pool, d1, d4], axis=-1, name='cat8')
        y = Activation('relu')(y)
        nt = Conv2D(ntp, (3, 3), padding='same', dilation_rate=8,trainable = False)(y)
        t = Conv2D(tp, (3, 3), padding='same', dilation_rate=8)(y)
        d8 = Concatenate(axis=3)([nt,t])
        
        y = concatenate([x,pool, d1, d4, d8], axis=-1, name='cat16')
        y = Activation('relu')(y)
        nt = Conv2D(ntp, (3, 3), padding='same', dilation_rate=16,trainable = False)(y)
        t = Conv2D(tp, (3, 3), padding='same', dilation_rate=16)(y)
        d16 = Concatenate(axis=3)([nt,t])
        
        x = concatenate([pool, d1, d4, d8, d16], axis=-1)
        x = InstanceNormalization()(x)
        x = Activation('relu')(x)
        x = SpatialDropout2D(0.25)(x)
        return x
    
    def initModel(self, dataset_name):
        assert dataset_name in ['CDnet', 'SBI', 'UCSD'], 'dataset_name must be either one in ["CDnet", "SBI", "UCSD"]]'
        assert len(self.img_shape)==3
        h, w, d = self.img_shape
        
        net_input = Input(shape=(h, w, d), name='net_input')
        res_output = self.Resnet(net_input)
        model = Model(inputs=net_input, outputs=res_output, name='model')
        
                
        x,a,b = model.output
        
                
        x = self.M_FPM(x)
        x,o4,o3,o2,o1 = self.decoder(x,a,b)
        
        # pad in case of CDnet2014
        if(self.scene=='tramCrossroad_1fps'):
            x = MyUpSampling2D(size=(1,1), num_pixels=(6,0), method_name=self.method_name)(x)
        elif(self.scene=='tunnelExit_0_35fps'):
            x = MyUpSampling2D(size=(1,1), num_pixels=(0,4), method_name=self.method_name)(x)
        elif(self.scene=='peopleInShade'):
            x = MyUpSampling2D(size=(1,1), num_pixels=(4,4), method_name=self.method_name)(x)
        elif(self.scene=='wetSnow'):
            x = MyUpSampling2D(size=(1,1), num_pixels=(4,0), method_name=self.method_name)(x)
        elif(self.scene=='busyBoulvard'):
            x = MyUpSampling2D(size=(1,1), num_pixels=(4,0), method_name=self.method_name)(x)
        elif(self.scene=='winterStreet'):
            x = MyUpSampling2D(size=(1,1), num_pixels=(4,0), method_name=self.method_name)(x)
        elif(self.scene=='fluidHighway'):
            x = MyUpSampling2D(size=(1,1), num_pixels=(2,4), method_name=self.method_name)(x)
        elif(self.scene=='streetCornerAtNight'):
            x = MyUpSampling2D(size=(1,1), num_pixels=(5,3), method_name=self.method_name)(x)
        elif(self.scene=='skating'):
            x = MyUpSampling2D(size=(1,1), num_pixels=(0,4), method_name=self.method_name)(x)
        elif(self.scene=='bridgeEntry'):
            x = MyUpSampling2D(size=(1,1), num_pixels=(6,6), method_name=self.method_name)(x)
        elif(self.scene=='fluidHighway'):
            x = MyUpSampling2D(size=(1,1), num_pixels=(6,0), method_name=self.method_name)(x)
        elif(self.scene=='streetCornerAtNight'): 
            x = MyUpSampling2D(size=(1,1), num_pixels=(1,0), method_name=self.method_name)(x)
            x = Cropping2D(cropping=((0, 0),(0, 1)))(x)
        elif(self.scene=='tramStation'):  
            x = Cropping2D(cropping=((1, 0),(0, 0)))(x)
            x = MyUpSampling2D(size=(1,1), num_pixels=(8,0), method_name=self.method_name)(x)
        elif(self.scene=='twoPositionPTZCam'):
            x = MyUpSampling2D(size=(1,1), num_pixels=(4,2), method_name=self.method_name)(x)
        elif(self.scene=='turbulence2'):
            x = Cropping2D(cropping=((1, 0),(0, 0)))(x)
            x = MyUpSampling2D(size=(1,1), num_pixels=(4,5), method_name=self.method_name)(x)
        elif(self.scene=='turbulence3'):
            x = MyUpSampling2D(size=(1,1), num_pixels=(6,0), method_name=self.method_name)(x)
                
        vision_model = Model(inputs=net_input, outputs=[x,o4,o3,o2,o1], name='vision_model')
        opt = keras.optimizers.RMSprop(lr = self.lr, rho=0.9, epsilon=1e-08, decay=0.)
        
        c_loss = loss
        c_acc = acc
        
        
        vision_model.compile(loss=loss, optimizer=opt, metrics=[c_acc])
        #vision_model.compile(loss=loss, optimizer=opt, metrics=[c_acc])
        return vision_model
