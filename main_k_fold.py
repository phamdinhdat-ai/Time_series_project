import pandas as pd 
import numpy as np 
import os 
import pickle
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
k_fold_static_path = 'data/5_fold_static'
k_fold_dynamic_path = 'data/5_fold_dynamic'
# label_names = ['Down', 'DownLEFT', 'DownRIGHT', 'Left', 'LeftDOWN', 'LeftUP', 'Right', 'RightDOWN', 'RightUP', 'Sit', 'Up', 'UpLEFT', 'UpRIGHT']

# def load_data(data_type):
#     # load data 
#     if data_type == 'static':# data with person have no actitity when sleep 
#         label_names = ['Down', 'DownLEFT', 'DownRIGHT', 'Left', 'LeftDOWN', 'LeftUP', 'Right', 'RightDOWN', 'RightUP', 'Sit', 'Up', 'UpLEFT', 'UpRIGHT']
#         train_dataset = np.load(f"{static_path}/static_train_set.npy")
#         val_dataset = np.load(f"{static_path}/static_val_set.npy")
#         test_dataset = np.load(f"{static_path}/static_test_set.npy")
#         return train_dataset, val_dataset, test_dataset, label_names
    
#     if data_type == 'dynamic':# data with person do some action gesture
#         label_names = ["Prone", "Lateral Left", "Lateral Right", "Supine"]
#         train_dataset = np.load(f"{dynamic_path}/dynamic_train_set.npy")
#         val_dataset = np.load(f"{dynamic_path}/dynamic_val_set.npy")
#         test_dataset = np.load(f"{dynamic_path}/dynamic_test_set.npy")
#         return train_dataset, val_dataset, test_dataset, label_names


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default="transformer", help='name of model training')
    parser.add_argument('--data_type', type=str, default="dynamic", help='name of dataset training')
    parser.add_argument('--num_classes', type=int, default=5, help='numbers of classes in the dataset')
    parser.add_argument('--epochs', type=int, default=100, help='epochs')
    parser.add_argument('--sequence_lenght', type=int, default=None, help='sequence_lenght for Sequence model')
    parser.add_argument('--overlap', type=float, default=None, help='overlap of window across samples')
    parser.add_argument('--batch_size', type=int, default=128, help='setting batch_size')
    parser.add_argument('--plot', type=bool, default=True, help='plot performance')
    return parser.parse_known_args()[0] if known else parser.parse_args()



opt = parse_opt(True)
data_type = opt.data_type
sequence_length = opt.sequence_lenght
overlap = opt.overlap

steps = int(sequence_length - overlap * sequence_length)
# set number of classes in each model
MLP_config['n_classes'] =  opt.num_classes 
CNN_config['n_classes'] =  opt.num_classes 
LSTM_config['n_classes'] =  opt.num_classes 
Transformer_config['n_classes'] =  opt.num_classes 


if data_type == 'k_fold_static':
    type_data = 'static'
    data_path = k_fold_static_path
    label_names = ['Down', 'DownLEFT', 'DownRIGHT', 'Left', 'LeftDOWN', 'LeftUP', 'Right', 'RightDOWN', 'RightUP', 'Sit', 'Up', 'UpLEFT', 'UpRIGHT']
elif data_type == 'k_fold_dynamic':
    data_path = k_fold_dynamic_path
    type_data = 'dynamic'
    label_names = ["Prone", "Lateral Left", "Lateral Right", "Supine"]
else: 
    KeyError("please type: k_fold_static or k_fold_dynamic")
if opt.sequence_lenght is not None:
    MLP_config['timestep'] = opt.sequence_lenght
    LSTM_config['timestep'] = opt.sequence_lenght
    Transformer_config['timestep'] = opt.sequence_lenght
    CNN_config["timestep"] = opt.sequence_lenght
else:
    MLP_config['timestep'] = 1
    LSTM_config['timestep'] = 1
    Transformer_config['timestep'] = 1
    CNN_config["timestep"] = 1

def main_k_fold(opt):
    epochs = opt.epochs
    batch_size = opt.batch_size
    model_type = opt.model_type
    plot = opt.plot
    if model_type == "transformer":
        model = Transformer()
        model.summary()

    if model_type == "lstm":
        model = LSTM()
        model.summary()

    if model_type == "cnn":
        model = CNN()
        model.summary()

    if model_type == "cnn_lstm":
        model = CNN_LSTM()
        model.summary()

    if model_type == "mlp":
        model = MLP()
        model.summary()
    else:
        print("enter correct name of model: transformer/lstm/cnn or mlp")

    if opt.data_type is not None:
        count = 1
        k_fold_results = []
        k_fold_hist = []
        for folder in os.listdir(data_path):
            print(f"Trainin {model_type} on {count}-fold {type_data} dataset: \n ")
            testset= np.load(f'{data_path}/{folder}/{type_data}_train_set.npy')
            trainset = np.load(f'{data_path}/{folder}/{type_data}_test_set.npy')
            train_data = trainset[:,1:4]
            train_label = trainset[:,4]
            test_data = testset[:,1:4]
            test_label = testset[:,4]
            
            # normalize data across min max scale
            train_scaled = min_max_scale(train_data)
            test_scaled = min_max_scale(test_data)
            # Category labels 
            train_y = to_categorical(train_label)
            test_y = to_categorical(test_label)

            if opt.sequence_lenght is not None:
                MLP_config['timestep'] = opt.sequence_lenght
                LSTM_config['timestep'] = opt.sequence_lenght
                Transformer_config['timestep'] = opt.sequence_lenght
                CNN_config["timestep"] = opt.sequence_lenght

                train_X, sequence_labels_train = generate_data(train_scaled, train_label, sequence_length= sequence_length, step= steps)
                test_X, sequence_labels_test = generate_data(test_scaled, test_label, sequence_length= sequence_length, step= steps)
                train_y = to_categorical(sequence_labels_train)
                test_y = to_categorical(sequence_labels_test)
            # samples = scaled_data.reshape(data.shape[0], 1, data.shape[1]).astype(dtype=np.float32)

            # train_X, test_X, train_y, test_y = train_test_split(samples, labeled, test_size= 0.2)
            else:
                MLP_config['timestep'] = 1
                LSTM_config['timestep'] = 1
                Transformer_config['timestep'] = 1
                CNN_config["timestep"] = 1
                train_y = to_categorical(train_label)
                test_y = to_categorical(test_label)

                train_X = train_scaled.reshape(train_scaled.shape[0], 1, train_scaled.shape[1]).astype(dtype=np.float32)
                test_X = test_scaled.reshape(test_scaled.shape[0], 1, test_scaled.shape[1]).astype(dtype=np.float32)

            print("shape of training data: ", train_X.shape)
            print("shape of testing data: ", test_X.shape)
            print("shape of training label: ", train_y.shape)
            print("shape of test label: ", test_y.shape)

            hist = model.train(train_X, train_y,test_X, test_y, epochs=epochs, batch_size=batch_size)
            result, y_pred  = model.evaluate(test_X, test_y)
            print('The evaluation of the model on the test set is: ', result)
            test_y_tf = np.argmax(test_y, axis=1)
            pred_y_tf = np.argmax(y_pred, axis=1)
            k_fold_results.append(result)
            k_fold_hist.append(hist.history)
            count +=1 
        with open(f'k_fold_results_{model_type}.pkl', 'wb') as f:
            pickle.dump(k_fold_hist, f)    
        k_fold_arr = np.array(k_fold_results)
        print(f"Average results on {type_data} dataset: ", np.mean(k_fold_arr, axis=0))
        print(f"Standard Deviation result on {type_data} dataset: ", np.std(k_fold_arr, axis=0))
    if plot == True:
        plot_performence(hist, epochs=epochs, model_name=model_type)
    #     # test_y_tf = np.argmax(test_y, axis=1)
    #     # pred_y_tf = np.argmax(y_pred, axis=1)
        plot_cf(pred_y_tf, test_y_tf, label_names=label_names, model_type=model_type)
   
if __name__ == "__main__":
    opt = parse_opt(True)
    main_k_fold(opt)
