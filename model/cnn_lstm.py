from datetime import datetime
from time import time
import json
import logging
import tensorflow as tf
import keras
from keras import layers
from keras.models import Model
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
import pickle
from keras.utils import plot_model 
from .config import LSTM_config

class CNN_LSTM(object):
    def __init__(self):
    
        self.timestep = 1, 
        self.n_features = 3, 
        self.n_classes  = 5,
        self.filters = [128, 64]
        self.kernel = 3, 
        self.mlp_units = [128, 64]
        self.dropout = 0.4
        self.log_dir = r"C:\Users\TAOSTORE\Desktop\SPP\checkpoint\log_dir/"
        self.save_file = r"C:\Users\TAOSTORE\Desktop\SPP\checkpoint\work_dir/"
        self.activation = "tanh"
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
        
        inputs = keras.Input(shape=(self.time_step, self.n_features))
        inputs_lstm = keras.Input(shape=(self.time_step, self.n_features))
        x = inputs
        for filter in self.filters:
            x =layers.Conv1D(filters=filter, kernel_size=1, activation='relu')(x)
            x = layers.MaxPool1D()(x)
        x = layers.Flatten()(x)
        y = inputs_lstm
        for hidden in self.hidden_size:
            y = layers.LSTM(units = hidden, activation = self.activation,return_sequences=True  )(y)
            # x = layers.MaxPooling(stride=3)(x)
            y = layers.Dropout(self.dropout)(y)

        y = layers.Flatten()(y)

        concate = layers.Concatenate()([x, y])
        for unit in self.mlp_units:
            concate = layers.Dense(unit, activation = 'tanh')(concate)
            concate = layers.Dropout(self.dropout)(concate)

        outs = layers.Dense(5, activation = "softmax")(concate)

        return Model([inputs, inputs_lstm], outs)
    

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
        self.filepath = self.save_file +"CNN_lstm_1D_2_5.best.hdf5"
        checkpoint = ModelCheckpoint(self.filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')

        callback_history = self.model.fit([X_train, X_train], y_train, epochs=epochs, batch_size=batch_size,
                             validation_split=0.2,
                             verbose=1,
                             callbacks=[early_stopping_monitor,checkpoint])
        
                             #callbacks=[PlotLossesKeras(), early_stopping_monitor, checkpoint])

        with open('results_cnn.pkl', 'wb') as f:
            pickle.dump(callback_history.history, f)  
        return callback_history

    def summary(self):
      self.model = self.build()
      plot_model(self.model)
      return self.model.summary()

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
        print("Classification Report: ", classification_report(pred_y_tf, test_y_tf),"\n")
        return result, y_pred
    