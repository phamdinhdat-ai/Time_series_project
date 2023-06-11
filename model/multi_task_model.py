
import os 
import pandas as pd 
import glob
import numpy as np
from keras.utils import plot_model
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
from model.config import Transformer_config, LSTM_config
from keras.layers import Layer
from keras import Model
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
        return context.reshape(context.shape[0],1)



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
        self.time_step = LSTM_config['timestep']
        self.n_features = LSTM_config['n_features']
        self.n_classes = LSTM_config['n_classes']
        self.hidden_size = LSTM_config['hidden_size']
        self.mlp_units = LSTM_config['mlp_units']
        self.dropout = LSTM_config['drop_out']
        self.log_dir = LSTM_config['log_dir']
        self.save_file = LSTM_config['save_file']
        self.activation = LSTM_config["activation"]
        self.width = 200
        self.height = 200 
        self.channel = 1

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
        
        # x = layers.LSTM(units = 256, activation = 'tanh',return_sequences=True , name = "Layers_LSTM_1" )(inputs)
        # x = layers.Dropout(self.dropout)(x)
        # x = layers.LSTM(units = 128, activation = 'tanh', return_sequences = True, name = "Layers_LSTM_2")(x)
        # x = layers.Dropout(self.dropout)(x)
        # input = keras.Input(shape=(self.time_step, self.n_features))

        x = inputs
        for hidden in self.hidden_size:
            x = layers.LSTM(units = hidden, activation = self.activation,return_sequences=True  )(x) 
            x = layers.Dropout(self.dropout)(x)

        # x = layers.Flatten()(x)
        # x = layers.Bidirectional(layers.LSTM(units = 64, activation = 'tanh',return_sequences=True , name = "Layers_BiLSTM_1"))(x)
        # x = layers.Bidirectional(layers.LSTM(units = 64, activation = 'tanh',return_sequences=True , name = "Layers_BiLSTM_2"))(x)
        x = layers.Flatten()(x)
        x = Attention()(x)
        # x = layers.Flatten()(x)
        # x = Attention()(x)
        x = layers.Dense(32)(x)
        return x
    def cnn_block(self, inputs):

        x = inputs 
        x = layers.Conv2D(32, (3, 3),activation = 'relu', )(x)
        x = layers.Conv2D(64, (3, 3),activation = 'relu', )(x)
        x = layers.Conv2D(64, (3, 3),activation = 'relu', )(x)
        x = layers.Conv2D(32, (3, 3),activation = 'relu', )(x)
        x = layers.Flatten()(x)
        x = layers.Dense(32)(x)
        return x 
    def build(self):
        input_signal = keras.Input(shape=(self.timestep, self.n_features))
        input_cnn = keras.Input(shape=(self.width, self.height, self.channel))
        
        out_att_lstm = self.attention_lstm(inputs=input_signal)
        out_cnn = self.cnn_block(inputs=input_cnn)
        concate = layers.Concatenate()([out_att_lstm, out_cnn])
        x = layers.Dropout(0.4)(concate)
        outputs = layers.Dense(5, activation='softmax')(x)
        return keras.Model([input_signal, input_cnn], outputs)
    def summary(self):
        self.model = self.build()
        plot_model(self.model)
        print(self.model.summary())
        