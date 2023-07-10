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

from model.cnn_lstm import CNN_LSTM

from preprocess.processing import min_max_scale, generate_data
from utils import plot_performence, plot_cf
from sklearn.metrics import classification_report

k_fold_static_path = 'data/5_fold_static_2'
k_fold_dynamic_path = 'data/5_fold_dynamic_2'
import json

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
    experiment = dict()
    if opt.data_type is not None:
        count = 1
        k_fold_results = []
        k_fold_hist = []
        for folder in os.listdir(data_path):
            print(f"Trainin {model_type} on {count}-fold {type_data} dataset: \n ")
            train_file = os.listdir(f"{data_path}/{folder}/train")[0]
            test_files = os.listdir(f"{data_path}/{folder}/test")
            _, ps_train = train_file.split('_')
            print("Training model on : ", ps_train[:-4])
            trainset = np.load(f'{data_path}/{folder}/train/{train_file}')
            train_data = trainset[:,1:4]
            train_label = trainset[:,4]
            # normalize data across min max scale
            train_scaled = min_max_scale(train_data)
            #load testset 
            test_list = []
            test_persons = []
            for file in test_files:
                _, name = file.split('_')
                data = np.load(f"{data_path}/{folder}/test/{file}")
                samples = data[:, 1:4]
                labels = data[:, 4]
                scaled_sample = min_max_scale(samples)
                test_list.append([scaled_sample,labels])
                test_persons.append(name)
            # normalize data across min max scale
            train_scaled = min_max_scale(train_data)
            # Category labels 

            if opt.sequence_lenght is not None:
                MLP_config['timestep'] = opt.sequence_lenght
                LSTM_config['timestep'] = opt.sequence_lenght
                Transformer_config['timestep'] = opt.sequence_lenght
                CNN_config["timestep"] = opt.sequence_lenght

                train_X, sequence_labels_train = generate_data(train_scaled, train_label, sequence_length= sequence_length, step= steps)
                train_y = to_categorical(sequence_labels_train)
                # split train data into 2 sets: training set/testing set (80/20)
                X_train, X_val, y_train, y_val = train_test_split(train_X, train_y, test_size=0.2)
                test_data = []
                for scaled_sample, labels in test_list:
                    X_test, sq_y = generate_data(scaled_sample, labels, sequence_length= sequence_length, step= steps)
                    y_test = to_categorical(sq_y)
                    test_data.append([X_test, y_test])

            # train_X, test_X, train_y, test_y = train_test_split(samples, labeled, test_size= 0.2)
            else:
                MLP_config['timestep'] = 1
                LSTM_config['timestep'] = 1
                Transformer_config['timestep'] = 1
                CNN_config["timestep"] = 1
                train_y = to_categorical(train_label)
                train_X = train_scaled.reshape(train_scaled.shape[0], 1, train_scaled.shape[1]).astype(dtype=np.float32)
                test_data = []
                for scaled_sample, labels in test_list:
                    y_test = to_categorical(labels)
                    X_test = scaled_sample.reshape(scaled_sample.shape[0], 1, scaled_sample.shape[1]).astype(dtype=np.float32)
                    test_data.append(X_test, y_test)
            print("shape of training data: ", X_train.shape)
            print("shape of training label: ", y_train.shape)
            print("shape of validation data: ", X_val.shape)
            print("shape of validation label: ", y_val.shape)
            # print("shape of testing data: ", test_X.shape)
            # print("shape of test label: ", test_y.shape)

            hist = model.train(X_train, y_train,X_val, y_val, epochs=epochs, batch_size=batch_size)
            
            experiment[f"train_{count}"] = ps_train

            #evaluation on test data 
            results = dict()
            cnt = 0
            for (X_test, y_test), name in zip(test_data, test_persons):
                print(f"The evaluation of the model on {name}'s data: ") 
                result, y_pred  = model.evaluate(X_test, y_test)
                print('The evaluation of the model: ', result)
                results[f"Person_{cnt}"] = name
                results[f"Evaluation_{cnt}"] = result 
                test_y_tf = np.argmax(y_test, axis=1)
                pred_y_tf = np.argmax(y_pred, axis=1)
                cnt +=1 
            print(results)
            experiment[f"test_result_{count}"] = results

            k_fold_results.append(result)
            k_fold_hist.append(hist.history)
            count +=1

        print(experiment)
        with open(f"k_fold_perform_{model_type}.pkl",'wb') as outfile:
            pickle.dump(experiment, outfile)
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
