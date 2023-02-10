
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

from .config import Transformer_config
class Transformer(object):
    """ Building the Recurrent Neural Network for Multivariate time series forecasting
    """

    def __init__(self):
        """ Initialization of the object
        """

        # Get model hyperparameters
        self.timestep = Transformer_config['timestep']
        self.n_features = Transformer_config['n_features']
        self.n_classes = Transformer_config['n_classes']

        self.log_dir = Transformer_config["log_dir"]
        self.checkpoint_dir = Transformer_config["save_file"]

        self.head_size = Transformer_config["head_size"]
        self.num_heads = Transformer_config["num_heads"]
        self.filters = Transformer_config["filter"]
        self.num_transformer_blocks = Transformer_config["num_encoder_blocks"]
        self.mlp_units = Transformer_config["mlp_units"]
        self.mlp_dropout=Transformer_config["drop_out"]
        self.dropout=Transformer_config["drop_out"]
        self.activation = Transformer_config["activation"]

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
        x = layers.Conv1D(filters=self.filters, kernel_size=1, activation='relu')(x)
        x = layers.Dropout(self.dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        return x + res


    def build(self):
        """ Build the model architecture
        """

        inputs = keras.Input(shape=(self.timestep, self.n_features))
        x = inputs
        for _ in range(self.num_transformer_blocks):
            x = self.transformer_encoder(x)

        # x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
        x = layers.Flatten()(x)
        for dim in self.mlp_units:
            x = layers.Dense(dim, activation=self.activation)(x)
            x = layers.Dropout(self.mlp_dropout)(x)

        # output layer
        outputs = layers.Dense(self.n_classes, activation='softmax')(x)

        return keras.Model(inputs, outputs)



    def train(self,
        X_train,
        y_train,
        epochs=200,
        batch_size=64):
        """ Training the network
        :param X_train: training feature vectors [#batch,#number_of_timesteps,#number_of_features]
        :type 3-D Numpy array of float values
        :param Y_train: training target vectors
        :type 2-D Numpy array of float values
        :param epochs: number of training epochs
        :type int
        :param batch_size: size of batches used at each forward/backward propagation
        :type int
        :return -
        :raises: -
        """

        self.model = self.build()
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                           loss = 'categorical_crossentropy',
                           metrics= ["acc",
                           tf.keras.metrics.Precision(),
                           tf.keras.metrics.Recall(),
                           tf.keras.metrics.AUC()]
                           )
        # print(self.model.summary())

        # Stop training if error does not improve within 50 iterations
        early_stopping_monitor = EarlyStopping(patience=50, restore_best_weights=True)

        # Save the best model ... with minimal error
        self.filepath = self.checkpoint_dir +"Transformer_6block_4head.best.hdf5"
        checkpoint = ModelCheckpoint(self.filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')

        callback_history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                             validation_split=0.2,
                             verbose=1,
                             callbacks=[early_stopping_monitor,checkpoint])
           
                             #callbacks=[PlotLossesKeras(), early_stopping_monitor, checkpoint])
        import pickle
        with open('results_transformer.pkl', 'wb') as f:
            pickle.dump(callback_history.history, f)  
        return callback_history
        # return callback_history

    def summary(self):
      self.model = self.build()
      print(self.model.summary())
    def evaluate(self,
        X_test,
        y_test):
        # y_pred = self.model.predict(X_test)
        # result = self.model.evaluate(X_test, y_test)
        from sklearn.metrics import classification_report 
        import numpy as np
        from keras.models import load_model
        model = load_model(self.filepath)
        result = model.evaluate(X_test, y_test)
        y_pred = model.predict(X_test)
        test_y_tf = np.argmax(y_test, axis=1)
        pred_y_tf = np.argmax(y_pred, axis=1)
        print("Classification Report:\n ", classification_report(pred_y_tf, test_y_tf),"\n")
        return result, y_pred

    

# model = Transformer()
# model.summary()