import os 
import glob 
import time 


import numpy as np 
import pandas as pd 
import pickle 

static_path = ".\data\STATIC_DYNAMIC_DATA_2\Data1_STATIC"
dynamic_path  = ".\data\STATIC_DYNAMIC_DATA_2\Data2_DYNAMIC"


static_classes = os.listdir(static_path)
dynamic_classes = os.listdir(dynamic_path)
count = 0
dataset = np.array([])
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
print(static_classes)
train_dataset = np.array([])
val_dataset = np.array([])
test_dataset = np.array([])
for folder in static_classes:
    print(f"=======Class: {folder}=======")
    files = os.listdir(f"{static_path}\{folder}")
    for file in files:
        print(file.split("_"))
        char = file.split('_')
        status, angle = char[3].lower(), char[-1].split('.')[0]
        df = pd.read_csv(f"{static_path}\{folder}\{files[0]}")
        df.columns = ["ID", "X", "Y", "Z"]
        df = df.dropna()
        df['Z']  = df['Z'].str.replace(';', '').astype(dtype=np.float32)
        print(status, folder.lower())
        if folder.lower() == 'sit':
            df = df[:3000]
            label = count*np.ones(df.shape[0])
        if folder.lower() == status:# change statis --> status[7:] for dynamic
            label = count*np.ones(df.shape[0])
        df['Labels'] = label

        if train_dataset.shape[0] == 0:
            # dataset = df.values
            train_dataset = df.values[:int(df.shape[0]*TRAIN_RATIO)]
            val_dataset  = df.values[int(df.shape[0]*TRAIN_RATIO):int(df.shape[0]*(TRAIN_RATIO + VAL_RATIO))]
            test_dataset = df.values[-int(df.shape[0]*TEST_RATIO) :]
        else:
            # dataset = np.concatenate([dataset, df.values], axis = 0)
            train_dataset = np.concatenate([train_dataset,df.values[:int(df.shape[0]*TRAIN_RATIO)]], axis = 0)
            val_dataset  = np.concatenate([val_dataset, df.values[int(df.shape[0]*TRAIN_RATIO):int(df.shape[0]*(TRAIN_RATIO + VAL_RATIO))]], axis = 0)
            test_dataset = np.concatenate([test_dataset, df.values[-int(df.shape[0]*TEST_RATIO) :]], axis = 0)

    # df.to_csv(file_save)
    count +=1
    print(count)

# np.save(".\data\static\static_train_dataset", train_dataset)
# np.save(".\data\static\static_val_dataset", val_dataset)
# np.save(".\data\static\static_test_dataset", test_dataset)
print("Shape of train dataset: ",train_dataset.shape)
print("Shape of validation dataset: ",val_dataset.shape) 
print("Shape of test dataset: ",test_dataset.shape) 

# static_dataset = pd.DataFrame(data=dataset, columns=["ID", "X", "Y", "Z", "Labels"])
# static_classes.to_csv("static_dataset.csv")