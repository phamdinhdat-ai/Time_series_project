import os 
import numpy as np
import pandas as pd 
from glob import glob 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical
import tensorflow as tf 

def load_and_process_data(file_path, sequence_lenght= 20, overlap = 0.3,  valid_ratio = None):
    dataset  = np.load(file=file_path)
    data = dataset[:,1:4]
    labels = dataset[:,4]
    steps = int(sequence_lenght - overlap * sequence_lenght)
    # normalize data across min max scale
    data_scaled = min_max_scale(data)
    X, sequence_labels = generate_data(data_scaled, labels, sequence_lenght= sequence_lenght, step=steps)
    y = to_categorical(sequence_labels)

    if valid_ratio is not None:
        X_train, X_val, y_train, y_val = train_test_split(X,y, test_size=valid_ratio)
        return X_train, X_val, y_train, y_val 
    return X, y

def tensorflow_dataset(data, labels, batch_size=64, shuffle=False, name='trainset'):
    tf_data = tf.data.Dataset.from_tensor_slices((data, labels))
    tf_data = tf_data.cache()
    # ds_train = ds_train.shuffle(buffer_size=100)
    tf_data = tf_data.batch(batch_size=batch_size, drop_remainder=True)
    tf_data = tf_data.prefetch(tf.data.AUTOTUNE)
    return tf_data


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



def generate_data(X, y, sequence_lenght = 10, step = 1):
    X_local = []
    y_local = []
    for start in range(0, X.shape[0] - sequence_lenght, step):
        end = start + sequence_lenght
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
    data_filtered = (1/data.shape[1])*np.sqrt(np.sum(data**2, axis=1))
    return data_filtered