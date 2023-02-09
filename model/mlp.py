from datetime import datetime
from time import time
import json
import logging
import tensorflow as tf
import keras
from keras import layers
from keras.models import Model
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint




class MLP(object):
    def __init__(self):
    
        self.timestep = 1, 
        self.n_features = 3, 
        self.n_classes  = 5,
        # self.filters = [128, 64]
        # self.kernel = 3, 
        self.mlp_units = [128, 64]
        self.dropout = 0.4
        self.log_dir = r"C:\Users\TAOSTORE\Desktop\SPP\checkpoint\log_dir_mlp/"
        self.save_file = r"C:\Users\TAOSTORE\Desktop\SPP\checkpoint\work_dir_mlp/"
        self.activation = "tanh"
        # self.filepath  = ''
    def build(self):
    
        inputs = keras.Input(shape=(1,3))
        x = inputs
        x = layers.Flatten()(x)
        for unit in self.mlp_units:
            x = layers.Dense(unit, activation = 'tanh')(x)
            x = layers.Dropout(self.dropout)(x)
        # x = layers.Flatten()(x)
        outs = layers.Dense(5, activation = "softmax")(x)

        return Model(inputs, outs)

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
        self.filepath = self.save_file +"MLP_1D_.best.hdf5"
        checkpoint = ModelCheckpoint(self.filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')

        callback_history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                             validation_split=0.2,
                             verbose=1,
                             callbacks=[early_stopping_monitor,checkpoint])
                             #callbacks=[PlotLossesKeras(), early_stopping_monitor, checkpoint])
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
    