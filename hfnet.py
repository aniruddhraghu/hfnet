import numpy as np

import sys
import os

import tensorflow as tf
from tensorflow import math as tfm


# set the random seed
SEED_VALUE = 0
os.environ['PYTHONHASHSEED']=str(SEED_VALUE)
tf.random.set_seed(SEED_VALUE)



class HFNet(tf.keras.Model) :
    def __init__(self,
        tab_size_spec,
        ecg_encoder,
        emb_dim = 16,
        tab_layers = [128], 
        new_layers = [256,128,64,2],   
        dout_rate = 0.0,
        act_f = tf.nn.relu, 
        training = True) :

        super(HFNet, self).__init__()

        self.tab_size_spec = tab_size_spec
        self.tablen = 2 # Hardcoded for now since we only have 2 tabular inputs
        
        # This is the ECG encoder which is loaded from a pretrained model, but is actually
        # retrained during HFNet learning
        self.ecg_encoder = ecg_encoder
        self.act_f = act_f
        
        # Batchnorm for the ECG encoder output
        self.bn0 = tf.keras.layers.BatchNormalization()
        
        # Batchnorm for the tabular features
        self.bn1 = tf.keras.layers.BatchNormalization()
        
        
        # Concat dropout
        self.dropout = tf.keras.layers.Dropout(dout_rate, seed=SEED_VALUE)
        
        # This is not super flexible, but is left over from a previous code version. A TODO is to update
        self.dropout_tab = tf.keras.layers.Dropout(dout_rate, seed=SEED_VALUE)
        self.dropout_tab2 = tf.keras.layers.Dropout(dout_rate, seed=SEED_VALUE)
        self.dropout_tab_li = [self.dropout_tab, self.dropout_tab2]


        # Project tabular features into a latent space
        self.emb_layers = {}
        for k in tab_size_spec.keys():
            self.emb_layers[k] = tf.keras.layers.Dense(emb_dim, activation=act_f)
        
        self.tab_layers = []
        self.dense_layers = []
        self.dropout_layers = []
        self.bns = []

        # No dropout if only inference
        if training==False :
            dout_rate = 0

        
        # Tabular encoder
        for d in tab_layers:
            self.tab_layers.append(tf.keras.layers.Dense(d))
        
        
        # Classifier dense blocks
        for i, d in enumerate(new_layers) :
            self.dense_layers.append(tf.keras.layers.Dense(d))
            if i < len(new_layers)-1:
                self.dropout_layers.append(tf.keras.layers.Dropout(dout_rate, seed=SEED_VALUE))
                self.bns.append(tf.keras.layers.BatchNormalization())


    def call(self, x) :
        xecg, xtab = x
        ecg = self.ecg_encoder(xecg)
        ecg = self.bn0(ecg)
        
        tab = xtab[:,:self.tablen]
        
        
        ## Tabular encoder
        # Get representations for the tabular features
        all_reprs = []
        for k,v in self.tab_size_spec.items():
            start = v[0]
            end = v[0] + v[1]
            feats = tab[:,start:end]
            rep = self.emb_layers[k](feats)
            all_reprs.append(rep)
        
        tab = tf.concat(all_reprs,axis=1)
        
        # Right now, this only supports a single tabular layer!
        for i in range(len(self.tab_layers)):
            tab = self.tab_layers[i](tab)
            tab = self.bn1(tab)
            tab = self.act_f(tab)
            tab = self.dropout_tab_li[i](tab)
        
        # Classifier
        y = tf.concat([ecg, tab], axis=1)
        y = self.dropout(y)
        for i in range(len(self.dense_layers)):
            y = self.dense_layers[i](y)
            if i < len(self.dense_layers)-1:
                y = self.bns[i](y)
                y = self.act_f(y)
                y = self.dropout_layers[i](y)
        return tf.keras.activations.softmax(y, axis=1)
