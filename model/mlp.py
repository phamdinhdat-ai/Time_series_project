from datetime import datetime
from time import time
import json
import logging
import tensorflow as tf
import keras
from keras import layers
from keras.models import Model
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from .config import MLP_config, Complier



class MLP(object):
    def __init__(self):
    
        self.timestep = MLP_config['timestep'] 
        self.n_features = MLP_config['n_features'] 
        self.n_classes  = MLP_config['n_classes']
        self.mlp_units = MLP_config['mlp_units']
        self.dropout = MLP_config['drop_out']
        self.log_dir =MLP_config['log_dir']
        self.save_file = MLP_config["save_file"]
        self.activation = MLP_config['activation']
        self.lr  = Complier['lr']
        self.optimizer  = Complier['optimizer'][0]
        self.loss_fn = Complier['loss']
    def build(self):
    
        inputs = keras.Input(shape=(self.timestep, self.n_features))
        x = inputs
        x = layers.Flatten()(x)
        for unit in self.mlp_units:
            x = layers.Dense(unit, activation = 'tanh')(x)
            x = layers.Dropout(self.dropout)(x)
        # x = layers.Flatten()(x)
        outs = layers.Dense(self.n_classes, activation = "softmax")(x)

        return Model(inputs, outs)

    def train(self,
        X_train,
        y_train,
        X_val, 
        y_val,
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
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
                           loss = self.loss_fn,
                           metrics= ["acc",
                           tf.keras.metrics.Precision(),
                           tf.keras.metrics.Recall(),
                           tf.keras.metrics.AUC()]
                           )
        # print(self.model.summary())

        # Stop training if error does not improve within 50 iterations
        early_stopping_monitor = EarlyStopping(patience=50, restore_best_weights=True)

        # Save the best model ... with minimal error
        self.filepath = self.save_file +"MLP_1D_.best.hdf5"
        checkpoint = ModelCheckpoint(self.filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')

        callback_history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                             validation_data = [X_val, y_val],
                             verbose=1,
                             callbacks=[early_stopping_monitor,checkpoint])
                             #callbacks=[PlotLossesKeras(), early_stopping_monitor, checkpoint])
        import pickle
        with open('mlp_result.pkl', 'wb') as f:
            pickle.dump(callback_history.history, f)
        return callback_history

    def summary(self):
      self.model = self.build()
      return self.model.summary()

    def evaluate(self,
        X_test,
        y_test):
        # y_pred = self.model.predict(X_test)
        # result = self.model.evaluate(X_test, y_test)
        from sklearn.metrics import classification_report 
        from keras.models import load_model
        model = load_model(self.filepath)
        result = model.evaluate(X_test, y_test)
        y_pred = model.predict(X_test)
        import numpy as np
        test_y_tf = np.argmax(y_test, axis=1)
        pred_y_tf = np.argmax(y_pred, axis=1)
        print("Classification Report:\n ", classification_report(pred_y_tf, test_y_tf),"\n")
        return result, y_pred
    