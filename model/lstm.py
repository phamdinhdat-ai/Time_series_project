
from datetime import datetime
from time import time
import json
import logging
import tensorflow as tf
import keras
from keras import layers
from keras.models import Model
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

from .config import LSTM_config
import pickle
class LSTM(object):


    def __init__(self) -> None:


        self.time_step = LSTM_config['timestep']
        self.n_features = LSTM_config['n_features']
        self.n_classes = LSTM_config['n_classes']
        self.hidden_size = LSTM_config['hidden_size']
        self.mlp_units = LSTM_config['mlp_units']
        self.dropout = LSTM_config['drop_out']
        self.log_dir = LSTM_config['log_dir']
        self.save_file = LSTM_config['save_file']
        self.activation = LSTM_config["activation"]
    def build(self):
        input = keras.Input(shape=(self.time_step, self.n_features))

        x = input
        for hidden in self.hidden_size:
            x = layers.LSTM(units = hidden, activation = self.activation,return_sequences=True  )(x)
            x = layers.Dropout(self.dropout)(x)

        x = layers.Flatten()(x)
        
        for unit in self.mlp_units:
            x = layers.Dense(unit, activation = self.activation)(x)
            x = layers.Dropout(self.dropout)(x)

        out = layers.Dense(self.n_classes, activation='softmax')(x)

        # x = layers.LSTM(units = 256, activation = 'tanh',return_sequences=True , name = "Layers_LSTM_1" )(input)
        # x = layers.Dropout(self.dropout)(x)
        # x = layers.LSTM(units = 128, activation = 'tanh', return_sequences = True, name  = "Layers_LSTM_2")(x)
        # x = layers.Flatten()(x)
        # x = layers.Dropout(self.dropout)(x)
        # x = layers.Dense(64)(x)
        # out = layers.Dense(self.n_classes, activation='softmax')(x)
        return  Model(input, out)
        # model.summary()
    def train(self,
        X_train,
        y_train, 
        epochs= 100,
        batch_size= 256):

        self.model = self.build()
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                           loss = 'categorical_crossentropy',
                           metrics= ["acc",
                           tf.keras.metrics.Precision(),
                           tf.keras.metrics.Recall(),
                           tf.keras.metrics.AUC()]
                           )
        early_stopping_monitor = EarlyStopping(patience=50, restore_best_weights=True)

        # Save the best model ... with minimal error
        self.filepath = self.save_file +"LSTM_2_Layers_2.best.hdf5"
        checkpoint = ModelCheckpoint(self.filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')

        callback_history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                             validation_split=0.2,
                             verbose=1,
                             callbacks=[early_stopping_monitor,checkpoint])
        with open('results_lstm.pkl', 'wb') as f:
            pickle.dump(callback_history.history, f) 
        return callback_history


    def summary(self):
        return self.build().summary()
    def evaluate(self,
            X_test,
            y_test):
            """ Evaluating the network
            :param X_test: test feature vectors [#batch,#number_of_timesteps,#number_of_features]
            :type 3-D Numpy array of float values
            :param Y_test: test target vectors
            :type 2-D Numpy array of int values
            :return  Evaluation losses
            :rtype 5 Float tuple
            :raise -
            """
            # y_pred = self.model.predict(X_test)
            # print(y_pred)
            # Print accuracy if ground truth is provided
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
            return  result,y_pred
    


    