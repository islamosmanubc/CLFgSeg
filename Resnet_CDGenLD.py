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
    
    def __init__(self, lr, img_shape, scene):
        self.lr = lr
        self.img_shape = img_shape
        self.scene = scene
        self.method_name = 'FgSegNet_v2'
        
    def Resnet(self, inp): 
        
        # Block 1
        x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format='channels_last')(inp)
        x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        xr = Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_convres', data_format='channels_last')(inp)
        x = add([x, xr])
        x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv3')(x)
        a = x
        x1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    
        # Block 2
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv1')(x1)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        xr = Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_convres')(x1)
        x = add([x, xr])
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv3')(x)
        b = x
        x2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    
        # Block 3
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv1')(x2)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        xr = Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_convres')(x2)
        x = add([x, xr])
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
        x3 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    
        # Block 4
        
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv1')(x3)
        x = Dropout(0.5, name='dr1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        xr = Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_convres')(x3)
        x = add([x, xr])
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        x = Dropout(0.5, name='dr2')(x)

        return x, a, b
    
    def decoder(self,x,a,b):
        a = GlobalAveragePooling2D()(a)
        b = Conv2D(64, (1, 1), strides=1, padding='same')(b)
        b = GlobalAveragePooling2D()(b)
        
        #block1
        x = Conv2D(64, (3, 3), strides=1, padding='same')(x)
        x = InstanceNormalization()(x)
        x = Activation('relu')(x)
        x1 = multiply([x, b])
        x = add([x, x1])
        x = UpSampling2D(size=(2, 2))(x)
        
        
        #block2
        x = Conv2D(32, (3, 3), strides=1, padding='same')(x)
        x = InstanceNormalization()(x)
        x = Activation('relu')(x)
        x2 = multiply([x, a])
        x = add([x, x2])
        xs2 = UpSampling2D(size=(2, 2))(x)

        
        #block3T1
        x = Conv2D(64, (3, 3), strides=1, padding='same')(xs2)
        x = InstanceNormalization()(x)
        x = Activation('relu')(x)
        x = UpSampling2D(size=(2, 2))(x)
        #block4T1
        
        
        o1 = Conv2D(2, (3,3), strides=1, padding='same', activation='sigmoid', name = 'block1out')(x)
        o2 = Conv2D(2, (3,3), strides=1, padding='same', activation='sigmoid', name = 'block2out')(x)
        o3 = Conv2D(2, (3,3), strides=1, padding='same', activation='sigmoid', name = 'block3out')(x)
        o4 = Conv2D(2, (3,3), strides=1, padding='same', activation='sigmoid', name = 'block4out')(x)
        o5 = Conv2D(2, (3,3), strides=1, padding='same', activation='sigmoid', name = 'block5out')(x)
        o6 = Conv2D(2, (3,3), strides=1, padding='same', activation='sigmoid', name = 'block6out')(x)
        
        c = Concatenate(axis=3)([x,o6,o5,o4,o3,o2,o1])
        x = Conv2D(32, (3, 3), activation='relu', padding='same', name='blockM_conv1')(c)
        d = Conv2D(1, (3,3), strides=1, padding='same', activation='sigmoid', name = 'merged')(x)
        return d,o6,o5,o4,o3,o2,o1
    
    def M_FPM(self, x):
        
        pool = MaxPooling2D((2, 2), strides=(1,1), padding='same')(x)
        pool = Conv2D(64, (1, 1), padding='same')(pool)
        
        d1 = Conv2D(64, (3, 3), padding='same')(x)
        
        y = concatenate([x, pool,d1], axis=-1, name='cat4')
        y = Activation('relu')(y)
        d4 = Conv2D(64, (3, 3), padding='same', dilation_rate=4)(y)
        
        y = concatenate([x, pool,d1, d4], axis=-1, name='cat8')
        y = Activation('relu')(y)
        d8 = Conv2D(64, (3, 3), padding='same', dilation_rate=8)(y)
        
        y = concatenate([x, pool,d1, d4,d8], axis=-1, name='cat16')
        y = Activation('relu')(y)
        d16 = Conv2D(64, (3, 3), padding='same', dilation_rate=16)(y)
        
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
        x,o6,o5,o4,o3,o2,o1 = self.decoder(x,a,b)
        

        vision_model = Model(inputs=net_input, outputs=[x,o6,o5,o4,o3,o2,o1], name='vision_model')
        opt = keras.optimizers.RMSprop(lr = self.lr, rho=0.9, epsilon=1e-08, decay=0.)
        
        c_loss = loss
        c_acc = acc
        
        vision_model.compile(loss=loss, optimizer=opt, metrics=[c_acc])
        #vision_model.compile(loss=c_loss, optimizer=opt, metrics=[c_acc])
        return vision_model
