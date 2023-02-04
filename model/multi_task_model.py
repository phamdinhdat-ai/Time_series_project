
import os 
import pandas as pd 
import glob
import numpy as np

from datetime import datetime
from time import time
import json
import logging

import tensorflow as tf
import keras
from keras import layers
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.callbacks import Callback
from keras.layers import Bidirectional 
from model.config import Transformer_config
from keras.layers import Layer
import keras.backend as K
class Attention(Layer):
    def __init__(self,**kwargs):
        super(Attention,self).__init__(**kwargs)
 
    def build(self,input_shape):
        self.W=self.add_weight(name='attention_weight', shape=(input_shape[-1],1), 
                               initializer='random_normal', trainable=True)
        self.b=self.add_weight(name='attention_bias', shape=(input_shape[1],1), 
                               initializer='zeros', trainable=True)        
        super(Attention, self).build(input_shape)
 
    def call(self,x):
        # Alignment scores. Pass them through tanh function
        e = K.tanh(K.dot(x,self.W)+self.b)
        # Remove dimension of size 1
        e = K.squeeze(e, axis=-1)   
        # Compute the weights
        alpha = K.softmax(e)
        # Reshape to tensorFlow format
        alpha = K.expand_dims(alpha, axis=-1)
        # Compute the context vector
        context = x * alpha
        context = K.sum(context, axis=1)
        return context



class SPPNet(object):

    def __init__(self) -> None:
         

        self.timestep = Transformer_config['timestep']
        self.n_features = Transformer_config['n_features']
        self.horizon = Transformer_config['n_classes']

        self.log_dir = Transformer_config["log_dir"]
        self.checkpoint_dir = Transformer_config["save_file"]

        self.head_size = Transformer_config["head_size"]
        self.num_heads = Transformer_config["num_heads"]
        self.filters = Transformer_config["filter"]
        self.num_transformer_blocks = Transformer_config["num_encoder_blocks"]
        self.mlp_units = Transformer_config["mlp_units"]
        self.mlp_dropout=Transformer_config["drop_out"]
        self.dropout=Transformer_config["drop_out"]

    def transformer_encoder(self,
    inputs):

        # Normalization and Attention
        x = layers.LayerNormalization(epsilon=1e-6)(inputs)
        x = layers.MultiHeadAttention(
        key_dim=self.head_size, num_heads=self.num_heads, dropout=self.dropout)(x, x)
        x = layers.Dropout(self.dropout)(x)

        res = x + inputs

        # Feed Forward Part
        x = layers.LayerNormalization(epsilon=1e-6)(res)
        x = layers.Conv1D(filters=self.filters, kernel_size=1, activation="relu")(x)
        x = layers.Dropout(self.dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        return x + res
    

    def attention_lstm(self, inputs):
        
        x = layers.LSTM(units = 256, activation = 'tanh',return_sequences=True , name = "Layers_LSTM_1" )(inputs)
        x = layers.Dropout(self.dropout)(x)
        x = layers.LSTM(units = 128, activation = 'tanh', return_sequences = False, name = "Layers_LSTM_2")(x)
        x = layers.Dropout(self.dropout)(x)

        x = layers.Bidirectional(layers.LSTM(units = 64, activation = 'tanh',return_sequences=True , name = "Layers_BiLSTM_1"))(x)
        x = layers.Bidirectional(layers.LSTM(units = 64, activation = 'tanh',return_sequences=True , name = "Layers_BiLSTM_2"))(x)
        x = layers.Flatten()(x)
        x = Attention()(x)
        x = layers.Dense(32)(x)
    def build_multi_task(self):
        
        return 0
