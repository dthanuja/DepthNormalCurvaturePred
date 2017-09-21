#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 18:20:20 2017

@author: thanuja
"""
from __future__ import print_function



import numpy as np
import theano
import theano.tensor as T
import lasagne.layers
import lasagne.layers.dnn
import lasagne.nonlinearities as NL
import lasagne
from PIL import Image

import matplotlib.cm as cm
from PIL import Image
import matplotlib.pyplot as plt

import cPickle
from collections import OrderedDict
from scipy.io import loadmat

xx = np.newaxis
net = {}

###################Principal Curvature Visualisation##########################
def get_color(K1,K2,val) :
    
    if(val <0):
        return np.zeros([1,3],dtype='double');
    else :
        x = K1*100.0;
        y = K2*100.0;
        
        if(y>x):
            temp =x
            x = y
            y= temp
                   
        x = x*1e-3;
        y = y*1e-3;

        color_out = np.zeros([3],dtype='double')
        if(x >= 0 and y >= 0):
            color_out[0] = 1
            color_out[1] = 1-x*y
            color_out[2] = 1-x*x-y*y;
        elif(x >= 0 and y < 0):
            color_out[0] = 1-y*y			         
            color_out[1] = 1
            color_out[2] = 1-x*x;
        elif(x < 0 and y < 0):
            color_out[0] = 1-x*x	-y*y		         
            color_out[1] = 1-x*y
            color_out[2] = 1;
        elif(x < 0 and y >= 0):
            color_out[0] = 1-x*x			         
            color_out[1] = 1
            color_out[2] = 1-y*y;
    
        color_out[color_out>1] = 1;
        color_out[color_out<0] = 0;    
         
        return color_out;
    
#####################Required Layer Functionalities###########################  
class Deconv2DLayer(lasagne.layers.Layer):

    def __init__(self, incoming, num_filters, filter_size, stride=1, pad=0,
            nonlinearity=lasagne.nonlinearities.rectify, **kwargs):
        super(Deconv2DLayer, self).__init__(incoming, **kwargs)
        self.num_filters = num_filters
        self.filter_size = lasagne.utils.as_tuple(filter_size, 2, int)
        self.stride = lasagne.utils.as_tuple(stride, 2, int)
        self.pad = lasagne.utils.as_tuple(pad, 2, int)
        self.W = self.add_param(lasagne.init.GlorotNormal(gain=1.0),
                (self.input_shape[1], num_filters) + self.filter_size,
                name='W')
        self.b = self.add_param(lasagne.init.Constant(0),
                (num_filters,),
                name='b')
        if nonlinearity is None:
            nonlinearity = lasagne.nonlinearities.identity
        self.nonlinearity = nonlinearity

    def get_output_shape_for(self, input_shape):
        shape = tuple(i*s - 2*p + f - 1
                for i, s, p, f in zip(input_shape[2:],
                                      self.stride,
                                      self.pad,
                                      self.filter_size))
        return (input_shape[0], self.num_filters) + shape

    def get_output_for(self, input, **kwargs):
        op = T.nnet.abstract_conv.AbstractConv2d_gradInputs(
            imshp=self.output_shape,
            kshp=(self.input_shape[1], self.num_filters) + self.filter_size,
            subsample=self.stride, border_mode=self.pad,filter_flip = False)
        conved = op(self.W, input, self.output_shape[2:])
        if self.b is not None:
            conved +=  self.b.reshape((1, self.b.shape[0], 1, 1))
        return (conved)

class NormLayer(lasagne.layers.Layer):
    def __init__(self, incoming, **kwargs):
        super(NormLayer, self).__init__(incoming, **kwargs)
    def get_output_shape_for(self, input_shape):
        return input_shape
    def get_output_for(self, input, **kwargs):
        scale = T.sqrt(T.sum(T.sqr(input),axis=1))
        return input/scale[:,None,:,:]           
        

    
def upsample(layer, nb_kernels,stride,bsize):
    xx = np.newaxis
    kx = np.linspace(0, 1, stride + 1)[1:-1]
    kx = np.concatenate((kx, [1], kx[::-1]))
    ker = kx[xx,:] * kx[:, xx]
    ker = T.constant(ker[xx,xx,:,:].astype(np.float32))
    xbatch = lasagne.layers.ReshapeLayer(layer,(layer.shape[0] * layer.shape[1], 1, layer.shape[2], layer.shape[3]))
 
    xup =  lasagne.layers.TransposedConv2DLayer(incoming=xbatch, num_filters=1, filter_size=7, stride=stride,crop='valid', W=ker, b=None, nonlinearity=None) 
    return lasagne.layers.ReshapeLayer(xup,(layer.shape[0], layer.shape[1], 59, 79))

###########################Load Weights from trained model#####################
def load_network(filename):
    f = file(filename, 'rb')
    param_values = cPickle.load(f)
    f.close()
    return param_values

###############################Build Network###################################   
def build_net(bsize,input_var=None):
     
     input_var2 =  input_var-np.array((123.68, 116.779, 103.939), dtype=np.float32)[xx,:,xx,xx]
     input_var3 = (input_var-109.31410628)/76.18328376
 
     net['input']  = lasagne.layers.InputLayer(shape=(bsize, 3, 228, 304),
                                        input_var=input_var2)
     
     net['input2']  = lasagne.layers.InputLayer(shape=(bsize, 3, 228, 304),
                                        input_var=input_var3)
   
    
     net['imnet_conv1_1'] = lasagne.layers.dnn.Conv2DDNNLayer(net['input'],num_filters=64,filter_size=(3,3),stride=(1,1),
                                                 pad=(0,0),W=lasagne.init.Constant(0.),b=lasagne.init.Constant(0.),flip_filters=False,nonlinearity=None)
     
     net['imnet_conv1_1'].W.tag.grad_scale=0
     net['imnet_conv1_1'].b.tag.grad_scale=0
     net['imnet_relu1_1']  = lasagne.layers.NonlinearityLayer(net['imnet_conv1_1'], nonlinearity=lasagne.nonlinearities.rectify)
     
     net['imnet_conv1_2'] = lasagne.layers.dnn.Conv2DDNNLayer(net['imnet_relu1_1'],num_filters=64,filter_size=(3,3),stride=(1,1),W=lasagne.init.Constant(0.),b=lasagne.init.Constant(0.),
                                                 flip_filters=False,nonlinearity=None)
     net['imnet_conv1_2'].W.tag.grad_scale=0
     net['imnet_conv1_2'].b.tag.grad_scale=0
     net['imnet_pool1']  = lasagne.layers.dnn.MaxPool2DDNNLayer( net['imnet_conv1_2'],pool_size=(2,2),stride=2)
     net['imnet_relu1']  = lasagne.layers.NonlinearityLayer(net['imnet_pool1'], nonlinearity=lasagne.nonlinearities.rectify)
   
     net['imnet_conv2_1'] = lasagne.layers.dnn.Conv2DDNNLayer(net['imnet_relu1'],num_filters=128,filter_size=(3,3),stride=(1,1),
                                                 pad=(1,1),flip_filters=False, nonlinearity=lasagne.nonlinearities.rectify)
     net['imnet_conv2_1'].W.tag.grad_scale=0
     net['imnet_conv2_1'].b.tag.grad_scale=0
     
     net['imnet_conv2_2'] = lasagne.layers.dnn.Conv2DDNNLayer(net['imnet_conv2_1'],num_filters=128,filter_size=(3,3),stride=(1,1),
                                                 pad=(1,1),flip_filters=False,nonlinearity=None)
     net['imnet_conv2_2'].W.tag.grad_scale=0
     net['imnet_conv2_2'].b.tag.grad_scale=0
     net['imnet_pool2'] = lasagne.layers.dnn.MaxPool2DDNNLayer(net['imnet_conv2_2'],pool_size=(2,2),stride=2)
     net['imnet_relu2']  = lasagne.layers.NonlinearityLayer(net['imnet_pool2'], nonlinearity=lasagne.nonlinearities.rectify)
     
     net['imnet_conv3_1'] = lasagne.layers.dnn.Conv2DDNNLayer(net['imnet_relu2'],num_filters=256,filter_size=(3,3),stride=(1,1),
                                                 pad=(1,1),flip_filters=False, nonlinearity=lasagne.nonlinearities.rectify)
     net['imnet_conv3_1'].W.tag.grad_scale=0
     net['imnet_conv3_1'].b.tag.grad_scale=0
     
     net['imnet_conv3_2'] = lasagne.layers.dnn.Conv2DDNNLayer(net['imnet_conv3_1'],num_filters=256,filter_size=(3,3),stride=(1,1),
                                                 pad=(1,1),flip_filters=False,nonlinearity=lasagne.nonlinearities.rectify)
     net['imnet_conv3_2'].W.tag.grad_scale=0
     net['imnet_conv3_2'].b.tag.grad_scale=0
     
     net['imnet_conv3_3'] = lasagne.layers.dnn.Conv2DDNNLayer(net['imnet_conv3_2'],num_filters=256,filter_size=(3,3),stride=(1,1),
                                                 pad=(1,1),flip_filters=False,nonlinearity=None)
     net['imnet_conv3_3'].W.tag.grad_scale=0
     net['imnet_conv3_3'].b.tag.grad_scale=0
     
     net['imnet_pool3'] = lasagne.layers.dnn.MaxPool2DDNNLayer(net['imnet_conv3_3'],pool_size=(2,2),stride=2)
     net['imnet_relu3']  = lasagne.layers.NonlinearityLayer(net['imnet_pool3'], nonlinearity=lasagne.nonlinearities.rectify)
     
     net['imnet_conv4_1'] = lasagne.layers.dnn.Conv2DDNNLayer(net['imnet_relu3'],num_filters=512,filter_size=(3,3),stride=(1,1),
                                                 pad=(1,1),flip_filters=False, nonlinearity=lasagne.nonlinearities.rectify)
     net['imnet_conv4_1'].W.tag.grad_scale=0
     net['imnet_conv4_1'].b.tag.grad_scale=0
     
     net['imnet_conv4_2'] = lasagne.layers.dnn.Conv2DDNNLayer(net['imnet_conv4_1'],num_filters=512,filter_size=(3,3),stride=(1,1),
                                                 pad=(1,1),flip_filters=False,nonlinearity=lasagne.nonlinearities.rectify)
     net['imnet_conv4_2'].W.tag.grad_scale=0
     net['imnet_conv4_2'].b.tag.grad_scale=0
     
     net['imnet_conv4_3'] = lasagne.layers.dnn.Conv2DDNNLayer(net['imnet_conv4_2'],num_filters=512,filter_size=(3,3),stride=(1,1),
                                                 pad=(1,1),flip_filters=False,nonlinearity=None)
     net['imnet_conv4_3'].W.tag.grad_scale=0
     net['imnet_conv4_3'].b.tag.grad_scale=0
     
     net['imnet_pool4'] = lasagne.layers.dnn.MaxPool2DDNNLayer(net['imnet_conv4_3'],pool_size=(2,2),stride=2)
     net['imnet_relu4']  = lasagne.layers.NonlinearityLayer(net['imnet_pool4'], nonlinearity=lasagne.nonlinearities.rectify)
     
     net['imnet_conv5_1'] = lasagne.layers.dnn.Conv2DDNNLayer(net['imnet_relu4'],num_filters=512,filter_size=(3,3),stride=(1,1),
                                                 pad=(1,1),flip_filters=False, nonlinearity=lasagne.nonlinearities.rectify)
     net['imnet_conv5_1'].W.tag.grad_scale=0
     net['imnet_conv5_1'].b.tag.grad_scale=0
     net['imnet_conv5_2'] = lasagne.layers.dnn.Conv2DDNNLayer(net['imnet_conv5_1'],num_filters=512,filter_size=(3,3),stride=(1,1),
                                                 pad=(1,1),flip_filters=False,nonlinearity=lasagne.nonlinearities.rectify)
     net['imnet_conv5_2'].W.tag.grad_scale=0
     net['imnet_conv5_2'].b.tag.grad_scale=0
     
     net['imnet_conv5_3'] = lasagne.layers.dnn.Conv2DDNNLayer(net['imnet_conv5_2'],num_filters=512,filter_size=(3,3),stride=(1,1),
                                                 pad=(1,1),flip_filters=False,nonlinearity=None)
     net['imnet_conv5_3'].W.tag.grad_scale=0
     net['imnet_conv5_3'].b.tag.grad_scale=0
     
     net['imnet_pool5'] = lasagne.layers.dnn.MaxPool2DDNNLayer(net['imnet_conv5_3'],pool_size=(2,2),stride=2)
     net['imnet_pool5']  = lasagne.layers.NonlinearityLayer(net['imnet_pool5'], nonlinearity=lasagne.nonlinearities.rectify)   
     net['imnet_pool5_scaled'] = lasagne.layers.ScaleLayer(net['imnet_pool5'] ,scales=lasagne.init.Constant(0.01))
     net['imnet_pool5_scaled'].scales.tag.grad_scale=0#  
     
     ##
     net['full1'] = lasagne.layers.DenseLayer(net['imnet_pool5_scaled'] ,num_units=4096,nonlinearity=lasagne.nonlinearities.rectify)
     net['full1'].W.tag.grad_scale=0
     net['full1'].b.tag.grad_scale=0
     net['full1scaled'] = lasagne.layers.ScaleLayer(net['full1'], scales=lasagne.init.Constant(0.5))
     net['full1scaled'].scales.tag.grad_scale=0
     net['full2'] = lasagne.layers.DenseLayer( net['full1scaled'] , num_units=17024,                                 
                                          nonlinearity=lasagne.nonlinearities.rectify)
     net['full2'].W.tag.grad_scale=0
     net['full2'].b.tag.grad_scale=0
     
     net['full2_reshaped']= lasagne.layers.ReshapeLayer(net['full2'],((bsize,64,14,19)))    
     net['full2_upsample']= upsample(net['full2_reshaped'],64,4,bsize)
   
     cropping = [None, None, 'center', 'center']       
     net['conv_s2_1'] = lasagne.layers.dnn.Conv2DDNNLayer(net['input2'],num_filters=96,filter_size=(9,9),stride=(2,2),
                                                 pad=(1,1),nonlinearity=lasagne.nonlinearities.rectify)  
     net['conv_s2_1'].W.tag.grad_scale=0
     net['conv_s2_1'].b.tag.grad_scale=0
     net['pool_s2_1']= lasagne.layers.dnn.MaxPool2DDNNLayer(net['conv_s2_1'],pool_size=(3,3),stride=2)     
     net['pool_s2_1_concat'] = lasagne.layers.ConcatLayer((net['full2_upsample'],net['pool_s2_1']),axis=1,cropping=cropping)
     
     ##Depths Scale 2
     net['depths_conv_new_s2_2'] = lasagne.layers.dnn.Conv2DDNNLayer(net['pool_s2_1_concat'],num_filters=64,filter_size=(5,5),stride=(1,1),
                                                 pad=(2,2),W=lasagne.init.HeNormal(gain='relu'), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify) 
     net['depths_conv_new_s2_3'] = lasagne.layers.dnn.Conv2DDNNLayer(net['depths_conv_new_s2_2'],num_filters=64,filter_size=(5,5),stride=(1,1),
                                                 pad=(2,2),W=lasagne.init.HeNormal(gain='relu'), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify)   
     net['depths_conv_new_s2_4'] = lasagne.layers.dnn.Conv2DDNNLayer(net['depths_conv_new_s2_3'],num_filters=64,filter_size=(5,5),stride=(1,1),
                                                 pad=(2,2), W=lasagne.init.HeNormal(gain='relu'), b=lasagne.init.Constant(0.),nonlinearity=lasagne.nonlinearities.rectify)          
     net['depths_conv_new_s2_5'] = Deconv2DLayer(net['depths_conv_new_s2_4'], 1, filter_size=5, stride=1,pad=2,nonlinearity=None)     
     net['depths_conv_s2_5_reshaped']= lasagne.layers.ReshapeLayer(net['depths_conv_new_s2_5'],((bsize,4070)))
     net['depths_bias'] = lasagne.layers.DenseLayer( net['depths_conv_s2_5_reshaped']  ,num_units=4070,nonlinearity=None,W=lasagne.init.Constant(0.))
     net['depths_bias_reshaped']= lasagne.layers.ReshapeLayer(net['depths_bias'],((bsize,1,55,74))) 
     net['depths_coarse'] = lasagne.layers.ElemwiseSumLayer((net['depths_conv_new_s2_5'],net['depths_bias_reshaped']))
     net['depths_coarse_up'] = lasagne.layers.Upscale2DLayer(net['depths_coarse'],2)
     net['depths_coarse_up_c'] = lasagne.layers.SliceLayer(net['depths_coarse_up'],indices=slice(0, 109),axis=2)
     net['depths_coarse_up_c'] = lasagne.layers.SliceLayer(net['depths_coarse_up_c'],indices=slice(0, 147),axis=3)
          
      ##Curvature Scale 2
     net['curvature_conv_s2_2'] = lasagne.layers.dnn.Conv2DDNNLayer(net['pool_s2_1_concat'],num_filters=64,filter_size=(5,5),stride=(1,1),
                                                 pad=(2,2),W=lasagne.init.HeNormal(gain='relu'), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify)    
     net['curvature_conv_s2_3'] = lasagne.layers.dnn.Conv2DDNNLayer(net['curvature_conv_s2_2'],num_filters=64,filter_size=(5,5),stride=(1,1),
                                                 pad=(2,2),W=lasagne.init.HeNormal(gain='relu'), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify)   
     net['curvature_conv_s2_4'] = lasagne.layers.dnn.Conv2DDNNLayer(net['curvature_conv_s2_3'],num_filters=64,filter_size=(5,5),stride=(1,1),
                                                 pad=(2,2), W=lasagne.init.HeNormal(gain='relu'), b=lasagne.init.Constant(0.),nonlinearity=lasagne.nonlinearities.rectify)        
     net['curvature_pred'] = lasagne.layers.dnn.Conv2DDNNLayer(net['curvature_conv_s2_4'],num_filters=2,filter_size=(5,5),stride=(1,1),
                                                 pad=(2,2),nonlinearity=None)     
     net['curvature_pred_up'] = lasagne.layers.Upscale2DLayer(net['curvature_pred'],2)
     net['curvature_pred_up_c'] = lasagne.layers.SliceLayer(net['curvature_pred_up'],indices=slice(0, 109),axis=2)
     net['curvature_pred_up_c'] = lasagne.layers.SliceLayer(net['curvature_pred_up_c'],indices=slice(0, 147),axis=3)
     
     ##Normals Scale 2
     net['normals_conv_new_s2_2'] = lasagne.layers.dnn.Conv2DDNNLayer(net['pool_s2_1_concat'],num_filters=64,filter_size=(5,5),stride=(1,1),
                                                 pad=(2,2), nonlinearity=lasagne.nonlinearities.rectify)
     net['normals_conv_new_s2_3'] = lasagne.layers.dnn.Conv2DDNNLayer(net['normals_conv_new_s2_2'],num_filters=64,filter_size=(5,5),stride=(1,1),
                                                 pad=(2,2), nonlinearity=lasagne.nonlinearities.rectify)       
     net['normals_conv_new_s2_4'] = lasagne.layers.dnn.Conv2DDNNLayer(net['normals_conv_new_s2_3'],num_filters=64,filter_size=(5,5),stride=(1,1),
                                                 pad=(2,2), nonlinearity=lasagne.nonlinearities.rectify)     
     net['normals_conv_new_s2_5'] = Deconv2DLayer(net['normals_conv_new_s2_4'], 3, filter_size=5, stride=1,pad=2,nonlinearity=None) 
     net['normals_conv_s2_5_normalised'] =NormLayer( net['normals_conv_new_s2_5'])    
     net['normals_coarse_up'] = lasagne.layers.Upscale2DLayer(net['normals_conv_s2_5_normalised'],2)
     net['normals_coarse_up_c'] = lasagne.layers.SliceLayer(net['normals_coarse_up'],indices=slice(0, 109),axis=2)
     net['normals_coarse_up_c'] = lasagne.layers.SliceLayer(net['normals_coarse_up_c'],indices=slice(0, 147),axis=3)
       
     
     net['conv_s3_1_new'] = lasagne.layers.dnn.Conv2DDNNLayer(net['input2'],num_filters=58,filter_size=(9,9),stride=(2,2),
                                                 pad=(1,1),W=lasagne.init.HeNormal(gain='relu'), b=lasagne.init.Constant(0.),nonlinearity=lasagne.nonlinearities.rectify)
     net['pool_s3_1']= lasagne.layers.dnn.MaxPool2DDNNLayer(net['conv_s3_1_new'],pool_size=(3,3),stride=1)
     net['pool_s3_1_concat'] = lasagne.layers.ConcatLayer((net['depths_coarse_up_c'],net['normals_coarse_up_c'], net['curvature_pred_up_c'],net['pool_s3_1']),axis=1)
     
     ##Depths Scale 3
     net['depths_conv_s3_2_new'] = lasagne.layers.dnn.Conv2DDNNLayer(net['pool_s3_1_concat'],num_filters=64,filter_size=(5,5),stride=(1,1),
                                                 pad=(2,2),W=lasagne.init.HeNormal(gain='relu'), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify)
     net['depths_conv_s3_3_new'] = lasagne.layers.dnn.Conv2DDNNLayer(net['depths_conv_s3_2_new'],num_filters=64,filter_size=(5,5),stride=(1,1),
                                                 pad=(2,2),W=lasagne.init.HeNormal(gain='relu'), b=lasagne.init.Constant(0.),nonlinearity=lasagne.nonlinearities.rectify)
     net['depths_conv_s3_4_new'] = Deconv2DLayer(net['depths_conv_s3_3_new'],1, filter_size=5, stride=1,pad=2,
                                nonlinearity=None)     

     ##Curvature Scale 3
     net['curvature_conv_s3_2'] = lasagne.layers.dnn.Conv2DDNNLayer(net['pool_s3_1_concat'],num_filters=64,filter_size=(3,3),stride=(1,1),
                                                 pad=(1,1),W=lasagne.init.HeNormal(gain='relu'), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify)
     net['curvature_conv_s3_3'] = lasagne.layers.dnn.Conv2DDNNLayer(net['curvature_conv_s3_2'],num_filters=64,filter_size=(5,5),stride=(1,1),
                                                 pad=(2,2),W=lasagne.init.HeNormal(gain='relu'), b=lasagne.init.Constant(0.),nonlinearity=lasagne.nonlinearities.rectify)
     net['curvature_conv_s3_4'] = Deconv2DLayer(net['curvature_conv_s3_3'],2, filter_size=5, stride=1,pad=2,
                                nonlinearity=None)     

     ##Normals Scale 3
     net['normals_conv_s3_2_new'] = lasagne.layers.dnn.Conv2DDNNLayer(net['pool_s3_1_concat'],num_filters=64,filter_size=(5,5),stride=(1,1),
                                                 pad=(2,2),W=lasagne.init.HeNormal(gain='relu'), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify)
     net['normals_conv_s3_3_new'] = lasagne.layers.dnn.Conv2DDNNLayer(net['normals_conv_s3_2_new'],num_filters=64,filter_size=(5,5),stride=(1,1),
                                                 pad=(2,2),W=lasagne.init.HeNormal(gain='relu'), b=lasagne.init.Constant(0.),nonlinearity=lasagne.nonlinearities.rectify)
     net['normals_conv_s3_4_new'] = Deconv2DLayer(net['normals_conv_s3_3_new'],3, filter_size=5, stride=1,pad=2,
                                nonlinearity=None)     
     net['normals_conv_s3_4_new_normalised'] =NormLayer( net['normals_conv_s3_4_new'])    


     return net
        
def test():
    
    n = 1
    input_var = T.tensor4('inputs')
    net = build_net(input_var=input_var,bsize=n);
 
    print('Loading model')
    saved_params = load_network('dnc.npy'); 
    lasagne.layers.set_all_param_values([net['depths_conv_s3_4_new'],net['curvature_conv_s3_4'],net['normals_conv_s3_4_new_normalised']], saved_params)
    
    depth = lasagne.layers.get_output(net['depths_conv_s3_4_new'])
    curvature = lasagne.layers.get_output(net['curvature_conv_s3_4'])
    normals =  lasagne.layers.get_output(net['normals_conv_s3_4_new_normalised'])

    print('Predicting')
    im = Image.open('test.png')
    im = im.resize((320, 240), Image.BICUBIC)
    im = np.asarray(im).reshape((1, 240, 320, 3))
    im = im.transpose((0,3,1,2))
    
    infer2 = theano.function([input_var], [depth,curvature,normals])
    depth_,curvature_,normals_ = infer2(im[:,:,6:234,8:312])
    for itr in range(0,1):
        
            output = np.squeeze(depth_[itr])
            logdepths_std =0.45723134;
            output = np.exp(output*logdepths_std);
            plt.figure()
            plt.imshow(output,cmap=cm.jet)
            plt.figure()
#            plt.imshow(im2[itr,:,:,:].transpose((1,2,0)))
            plt.imshow(im[0,:,:,:].transpose((1,2,0)))
            c1 = np.zeros([109,147,3],dtype='double'); 
            val1 =np.ones([109,147,1],dtype='double');  
            K1 = (curvature_[itr][0])
            K2 = (curvature_[itr][1])
            for i in range(0,109):
                for j in range(0,147):
                    c1[i][j][...] = get_color(K1[i][j]*10,K2[i][j]*10,val1[i][j]);
            plt.figure();
            plt.imshow(c1)
            
            c5 = np.zeros([109,147,3],dtype='double');

            i2 = normals_[itr][0]
            j2 = normals_[itr][1]
            k2 = normals_[itr][2]
            
            c5[:,:,0] = i2/2 +0.5;
            c5[:,:,1] = j2/2 +0.5;
            c5[:,:,2] = k2/2 +0.5;
            plt.figure()
            plt.imshow(c5)
            plt.show()


           
        
   
#            plt.colorbar()
        
if __name__ == '__main__':
   test()
