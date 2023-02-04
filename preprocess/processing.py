import pandas as pd 
import numpy as np 
import os 
from sklearn.preprocessing import MinMaxScaler


def load_data(file_path):
    df = pd.read_csv(file_path)
    return df 

def load_regr_data(file_path): 
    data = []
    labels = []
    # i = 0
    list_data = os.listdir(file_path)
    for file in list_data:
        df = pd.read_csv(file_path + file)
        char = file.split('_')
        rate, status  = int(char[3]), char[-1].split('.')[0]
        df.columns = ["ID", "X", "Y", "Z"]
        df = df.dropna()
        df['Z']  = df['Z'].str.replace(';', '').astype(dtype=np.float32)
        df = df.drop(labels='ID', axis=1)
        # label = int(eval(file[-6:-4]))
        sample = df.to_numpy()[:3000]
        # i += 1
        print(file)
        if sample.shape[0] == 3000:
            data.append(sample)
            labels.append(rate)
        else: 
            pass
    return np.array(data), np.array(labels)

def generate_data(X, y, sequence_length = 10, step = 1):
    X_local = []
    y_local = []
    for start in range(0, X.shape[0] - sequence_length, step):
        end = start + sequence_length
        X_local.append(X[start:end])
        y_local.append(y[end-1])
    return np.array(X_local), np.array(y_local)

def min_max_scale(dataset):
    scaler = MinMaxScaler()
    scaler.fit(dataset)
    scaled_data = scaler.transform(dataset)
    return scaled_data

# add filter update 

def filter_norm(data):
    # data: shape(n_samples, [X, Y, Z])
    data_filtered = (1/data.shape[1])*np.sprt(np.sum(np.pow(data), axis=0))
    return data_filtered

