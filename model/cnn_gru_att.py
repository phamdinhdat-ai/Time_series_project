# from config.adaptive_lstm import Config
from datetime import datetime
from time import time
import json
import logging
import keras
import tensorflow as tf
from keras import layers
from keras.models import Model
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras import regularizers
import pickle
from keras import backend as K
from keras.layers.core import Permute, RepeatVector, Lambda
from keras.layers import Multiply
# from config.adaptive_lstm import Config

def attention_block(inputs, single_attention_vector = True):
        time_steps = K.int_shape(inputs)[1] # time step
        dim = K.int_shape(inputs)[2]
        a = Permute((2, 1))(inputs)
        a = layers.Dense(time_steps, activation='softmax')(a)
        if single_attention_vector:
                a = Lambda(lambda x: K.mean(x, axis = 1))(a)
                a = RepeatVector(dim)(a)
        a_probs = Permute((2, 1))(a)
        attention = Multiply()([inputs, a_probs])
        return attention

class CNNGRUAttention(keras.Model):
    def __init__(self, config):
        super(CNNGRUAttention, self).__init__()
        self.config = config
        self.timestep = self.config.timestep
        self.n_features = self.config.n_features
        self.n_classes  = self.config.n_classes 
        self.hidden_size = self.config.hidden_size
        self.kernel_size = self.config.kernel_size
        self.strides = self.config.strides
        self.filters = self.config.filters
        self.dropout = self.config.dropout 
        self.log_dir = self.config.log_dir
        self.save_file = self.config.save_file
        self.activation = self.config.activation
        self.regularizers = self.config.regularizers
        self.normalizer = self.config.normalizer
        self.lr  = self.config.lr
        self.optimizer = self.config.optimizer
        self.loss_fn = self.config.loss_fn
        self.normalizer = self.config.normalizer


        if self.config.regularizers == 'l1':
            self.regularizers = regularizers.L1(l1=0.1)
        elif self.config.regularizers == 'l2':
            self.regularizers = regularizers.L2(l2=0.1)
        elif self.config.regularizers == "l1_l2":
            self.regularizers = regularizers.L1L2(l1 = 0.1, l2=0.1)
        else:
            self.regularizers = None
    def build(self):
        input = keras.Input(shape=(self.timestep, self.n_features))
        x = input
        x = layers.Conv1D(filters=self.filters, kernel_size=self.kernel_size, strides=self.strides, padding='same')(x)
        if self.normalizer == "batch_norm":
                x = layers.BatchNormalization()(x)
        elif self.normalizer == "layer_norm":
                x = layers.LayerNormalization(axis= -1, center=True , scale=True)(x)
        elif self.normalizer == "norm":
                x = tf.keras.layers.Normalization()(x)
        else:
                pass  
        # x = layers.MaxPool1D(pool_size=3, padding='valid', strides=1)(x)
        
        for hidden in self.hidden_size:
                # x = layers.BatchNormalization()(x)
                x, h_state,  c_state = layers.Bidirectional(layers.GRU(units = hidden, activation = self.activation, return_sequences=True, kernel_regularizer=self.regularizers, return_state=True))(x)
                x = layers.BatchNormalization()(x)
                # x = layers.Reshape((hidden*2, 1), input_shape = (hidden*2, ))
                
        x = layers.MaxPool1D(pool_size=3, padding='valid', strides=1)(x)
        x = attention_block(x)
        # x = layers.BatchNormalization()(x)
        x = layers.Flatten()(x)
        x = layers.Dropout(self.dropout)(x)
        out = layers.Dense(self.n_classes, activation='softmax')(x)
        model = Model(inputs = input, outputs = out)

        return model 
