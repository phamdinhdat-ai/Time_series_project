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
# model = Transformer()
from preprocess.processing import min_max_scale, generate_data
from utils import plot_performence, plot_cf
from sklearn.metrics import classification_report
file_path  = r"C:\Users\TAOSTORE\Desktop\SPP\data\SPP_final_1.csv"
save_file = r"C:\Users\TAOSTORE\Desktop\SPP\result\cnn"
dataset = pd.read_csv(file_path)
data = dataset.loc[:,"X":"Z"].values
labels = dataset.Labels.values.astype(dtype=np.float32)






label_names = ["Lieleft", "LieRight", "LieUp","LieDown" ,"Unsleep"]
scaled_data = min_max_scale(data)
# sequence_data, sequence_labels = generate_data(scaled_data, labels)



samples = scaled_data.reshape(data.shape[0], 1, data.shape[1]).astype(dtype=np.float32)
labeled = to_categorical(labels)

train_X, test_X, train_y, test_y = train_test_split(samples, labeled, test_size= 0.2)


print("shape of training data: ", train_X.shape)
print("shape of testing data: ", test_X.shape)
print("shape of labels data: ", labeled.shape)

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default="transformer", help='name of model training')
    parser.add_argument('--epochs', type=int, default=100, help='epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='setting batch_size')
    parser.add_argument('--plot', type=bool, default=True, help='plot performance')
    return parser.parse_known_args()[0] if known else parser.parse_args()

def main(opt):
    epochs = opt.epochs
    batch_size = opt.batch_size
    model_type = opt.model_type
    plot = opt.plot
    if model_type == "transformer":
        model = Transformer()
        model.summary()
        hist = model.train(train_X, train_y, epochs=epochs, batch_size=batch_size)
        result, y_pred  = model.evaluate(test_X, test_y)
        print('The evaluation of the model on the test set is: ', result)
        test_y_tf = np.argmax(test_y, axis=1)
        pred_y_tf = np.argmax(y_pred, axis=1)
    if model_type == "lstm":
        model = LSTM()
        model.summary()
        hist = model.train(train_X, train_y, epochs=epochs, batch_size=batch_size)
        result, y_pred  = model.evaluate(test_X, test_y)
        print('The evaluation of the model on the test set is: ', result)
        test_y_tf = np.argmax(test_y, axis=1)
        print(y_pred)
        pred_y_tf = np.argmax(y_pred, axis=1)
    if model_type == "cnn":
        model = CNN()
        model.summary()
        hist = model.train(train_X, train_y, epochs=epochs, batch_size=batch_size)
        result, y_pred  = model.evaluate(test_X, test_y)
        print('The evaluation of the model on the test set is: ', result)
        test_y_tf = np.argmax(test_y, axis=1)
        pred_y_tf = np.argmax(y_pred, axis=1)
    if model_type == "mlp":
        model = MLP()
        model.summary()
        hist = model.train(train_X, train_y, epochs=epochs, batch_size=batch_size)
        result, y_pred  = model.evaluate(test_X, test_y)
        print('The evaluation of the model on the test set is: ', result)
        test_y_tf = np.argmax(test_y, axis=1)
        pred_y_tf = np.argmax(y_pred, axis=1)
    else:
        KeyError("enter correct name of model: transformer/lstm/cnn or mlp")

    if plot == True:
        plot_performence(hist, epochs=epochs, model_name=model_type)
        # test_y_tf = np.argmax(test_y, axis=1)
        # pred_y_tf = np.argmax(y_pred, axis=1)
        plot_cf(pred_y_tf, test_y_tf, label_names=label_names, model_type=model_type)


if __name__ == "__main__":
    opt = parse_opt(True)
    main(opt)
# file_path = r"C:\Users\TAOSTORE\Desktop\SPP\data\SPP_Data/"


# from preprocess.processing import load_regr_data

# data, labels = load_regr_data(file_path=file_path)


# print("shape of data: ",data.shape)
# print(labels)



