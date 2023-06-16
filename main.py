import pandas as pd 
import numpy as np 
import os 

import argparse
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from model.transformer import Transformer
from model.lstm import LSTM
from model.cnn import CNN
from model.mlp import MLP
from model.config import LSTM_config, Transformer_config, CNN_config, MLP_config
# from model import config 
# model = Transformer()
from model.cnn_lstm import CNN_LSTM

# model = CNN_LSTM()
# print(model.summary())

from preprocess.processing import min_max_scale, generate_data
from utils import plot_performence, plot_cf
from sklearn.metrics import classification_report
static_path = 'data\static'
dynamic_path = 'data\dynamic'
label_dynamic_names = ["Prone", "Lateral Left", "Lateral Right", "Supine"]
def load_data(data_type):
    # load data 
    if data_type == 'static':# data with person have no actitity when sleep 
        list_files = os.listdir(static_path)
        print(list_files)
        train_dataset = np.load(f"{static_path}\{list_files[1]}")
        val_dataset = np.load(f"{static_path}\{list_files[2]}")
        test_dataset = np.load(f"{static_path}\{list_files[0]}")
        return train_dataset, val_dataset, test_dataset
    
    if data_type == 'dynamic':# data with person do some action gesture
        list_files = os.listdir(dynamic_path)
        train_dataset = np.load(f"{dynamic_path}\{list_files[1]}")
        val_dataset = np.load(f"{dynamic_path}\{list_files[2]}")
        test_dataset = np.load(f"{dynamic_path}\{list_files[0]}")
        return train_dataset, val_dataset, test_dataset
    
def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default="transformer", help='name of model training')
    parser.add_argument('--data_type', type=str, default="dynamic", help='name of dataset training')
    parser.add_argument('--num_classes', type=int, default=5, help='numbers of classes in the dataset')
    parser.add_argument('--epochs', type=int, default=100, help='epochs')
    parser.add_argument('--sequence_lenght', type=int, default=None, help='sequence_lenght for Sequence model')
    parser.add_argument('--batch_size', type=int, default=128, help='setting batch_size')
    parser.add_argument('--plot', type=bool, default=True, help='plot performance')
    return parser.parse_known_args()[0] if known else parser.parse_args()




opt = parse_opt(True)
data_type = opt.data_type
sequence_length = opt.sequence_lenght

# set number of classes in each model
MLP_config['n_classes'] =  opt.num_classes 
CNN_config['n_classes'] =  opt.num_classes 
LSTM_config['n_classes'] =  opt.num_classes 
Transformer_config['n_classes'] =  opt.num_classes 


# split data into three sets
train_dataset, val_dataset, test_dataset = load_data(data_type)
print(val_dataset.shape)
train_data = train_dataset[:,1:4]
train_label = train_dataset[:,4]
val_data = val_dataset[:,1:4]
val_label = val_dataset[:,4]
test_data = test_dataset[:,1:4]
test_label = test_dataset[:,4]


# normalize data across min max scale
train_scaled = min_max_scale(train_data)
val_scaled = min_max_scale(val_data)
test_scaled = min_max_scale(test_data)

# Category labels 
train_y = to_categorical(train_label)
val_y = to_categorical(val_label)
test_y = to_categorical(test_label)


if opt.sequence_lenght is not None:
    MLP_config['timestep'] = opt.sequence_lenght
    LSTM_config['timestep'] = opt.sequence_lenght
    Transformer_config['timestep'] = opt.sequence_lenght
    CNN_config["timestep"] = opt.sequence_lenght

    train_X, sequence_labels_train = generate_data(train_scaled, train_label, sequence_length= sequence_length)
    val_X, sequence_labels_val = generate_data(val_scaled, val_label, sequence_length= sequence_length)
    test_X, sequence_labels_test = generate_data(test_scaled, test_label, sequence_length= sequence_length)
    train_y = to_categorical(sequence_labels_train)
    val_y = to_categorical(sequence_labels_val)
    test_y = to_categorical(sequence_labels_test)
# samples = scaled_data.reshape(data.shape[0], 1, data.shape[1]).astype(dtype=np.float32)

# train_X, test_X, train_y, test_y = train_test_split(samples, labeled, test_size= 0.2)
else:
    MLP_config['timestep'] = 1
    LSTM_config['timestep'] = 1
    Transformer_config['timestep'] = 1
    CNN_config["timestep"] = 1
    train_y = to_categorical(train_label)
    val_y = to_categorical(val_label)
    test_y = to_categorical(test_label)

    train_X = train_scaled.reshape(train_scaled.shape[0], 1, train_scaled.shape[1]).astype(dtype=np.float32)
    val_X = val_scaled.reshape(val_scaled.shape[0], 1, val_scaled.shape[1]).astype(dtype=np.float32)
    test_X = test_scaled.reshape(test_scaled.shape[0], 1, test_scaled.shape[1]).astype(dtype=np.float32)

print("shape of training data: ", train_X.shape)
print("shape of testing data: ", test_X.shape)
print("shape of training label: ", train_y.shape)


def main(opt):
    epochs = opt.epochs
    batch_size = opt.batch_size
    model_type = opt.model_type
    plot = opt.plot
    if model_type == "transformer":
        model = Transformer()
        model.summary()
        hist = model.train(train_X, train_y,val_X, val_y, epochs=epochs, batch_size=batch_size)
        result, y_pred  = model.evaluate(test_X, test_y)
        print('The evaluation of the model on the test set is: ', result)
        test_y_tf = np.argmax(test_y, axis=1)
        pred_y_tf = np.argmax(y_pred, axis=1)
    if model_type == "lstm":
        model = LSTM()
        model.summary()
        hist = model.train(train_X, train_y,val_X, val_y, epochs=epochs, batch_size=batch_size)
        result, y_pred  = model.evaluate(test_X, test_y)
        print('The evaluation of the model on the test set is: ', result)
        test_y_tf = np.argmax(test_y, axis=1)
        pred_y_tf = np.argmax(y_pred, axis=1)
    if model_type == "cnn":
        model = CNN()
        model.summary()
        hist = model.train(train_X, train_y,val_X, val_y, epochs=epochs, batch_size=batch_size)
        result, y_pred  = model.evaluate(test_X, test_y)
        print('The evaluation of the model on the test set is: ', result)
        test_y_tf = np.argmax(test_y, axis=1)
        pred_y_tf = np.argmax(y_pred, axis=1)
    if model_type == "cnn_lstm":
        model = CNN_LSTM()
        model.summary()
        hist = model.train(train_X, train_y,val_X, val_y, epochs=epochs, batch_size=batch_size)
        result, y_pred  = model.evaluate(test_X, test_y)
        print('The evaluation of the model on the test set is: ', result)
        test_y_tf = np.argmax(test_y, axis=1)
        pred_y_tf = np.argmax(y_pred, axis=1)
    if model_type == "mlp":
        model = MLP()
        model.summary()
        hist = model.train(train_X, train_y,val_X, val_y, epochs=epochs, batch_size=batch_size)
        result, y_pred  = model.evaluate(test_X, test_y)
        print('The evaluation of the model on the test set is: ', result)
        test_y_tf = np.argmax(test_y, axis=1)
        pred_y_tf = np.argmax(y_pred, axis=1)
    else:
        KeyError("enter correct name of model: transformer/lstm/cnn or mlp")

    if plot == True:
        plot_performence(hist, epochs=epochs, model_name=model_type)
    #     # test_y_tf = np.argmax(test_y, axis=1)
    #     # pred_y_tf = np.argmax(y_pred, axis=1)
        plot_cf(pred_y_tf, test_y_tf, label_names=label_names, model_type=model_type)


if __name__ == "__main__":
    opt = parse_opt(True)
    main(opt)
# file_path = r"C:\Users\TAOSTORE\Desktop\SPP\data\SPP_Data/"


# from preprocess.processing import load_regr_data

# data, labels = load_regr_data(file_path=file_path)

# print("shape of data: ",data.shape)
# print(labels)



# from model.multi_task_model import SPPNet
# from model.cnn_lstm import CNN_LSTM

# model = CNN_LSTM()
# print(model.summary())