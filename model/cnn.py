from datetime import datetime
from time import time
import json
import logging
import tensorflow as tf
import keras
from keras import layers
from keras.models import Model
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras import regularizers
from model.batchnorm  import BatchNorm
import pickle


class CNN(keras.Model):
    def __init__(self, config):
        super(CNN,self).__init__()

        self.config = config
        self.timestep = self.config.timestep
        self.n_features = self.config.n_features
        self.n_classes  = self.config.n_classes 
        self.filters = self.config.filters
        self.kernel_size = self.config.kernel_size
        self.mlp_units =  self.config.mlp_units
        self.dropout = self.config.dropout 
        self.log_dir = self.config.log_dir
        self.save_file = self.config.save_file
        self.activation = self.config.activation
        self.regularizers = self.config.regularizers
        self.normalizer = self.config.normalizer
        self.lr  = self.config.lr
        self.optimizer = self.config.optimizer
        self.loss_fn = self.config.loss_fn

        if self.config.normalizer == "batch norm":
            self.normalizer = layers.BatchNormalization()
        elif self.config.normalizer == "layer norm":
            self.normalizer = layers.LayerNormalization()
        else:
            self.normalizer = BatchNorm()


        if self.config.regularizers == 'l1':
            self.regularizers = regularizers.L1(l1=0.1)
        elif self.config.regularizers == 'l2':
            self.regularizers = regularizers.L2(l2=0.1)
        elif self.config.regularizers == "l1_l2":
            self.regularizers = regularizers.L1L2(l1 = 0.1, l2=0.1)
        else:
            self.regularizers = None
    def build(self):
    
        inputs = keras.Input(shape=(self.timestep,self.n_features))
        x = inputs
        for filter in self.filters:
            x =layers.Conv1D(filters=filter, kernel_size=1, activation='relu', kernel_regularizer=self.regularizers)(x)
            x = self.normalizer(x)
        x = layers.Flatten()(x)

        for unit in self.mlp_units:
            x = layers.Dense(unit, activation = 'tanh', kernel_regularizer=self.regularizers)(x)
            x = self.normalizer(x)
            x = layers.Dropout(self.dropout)(x)

        outs = layers.Dense(self.n_classes, activation = "softmax")(x)

        return Model(inputs, outs)
