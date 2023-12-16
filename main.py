import pickle
import os
import pandas as pd 
import tensorflow as tf
from model.lstm import LSTM
from model.adaptive_lstm import AdaptiveLSTM
from config.lstm import Config 
from config.lstm import Config
# from utils.trainer import train_model
from utils.load_data import load_and_process_data, tensorflow_dataset, load_from_folder
# from utils.trainer import test_model
from utils.parse import parse_opt
from utils.plot import plot_performance
from utils.trainer import train_model
from utils.losses import negative_log_likelihood
from utils.trainer import test_model
from datetime import date
from utils.trainer import experiment
import keras
print(tf.__version__)

today = str(date.today())
#load data
# train_path = 'data/dynamic1/train/train_PhamQuangTu.npy'

train_folder = 'data/new_data_static/trainset'
test_path = 'data/new_data_static/testset'
test_neck_path = 'data/necktest'
opt = parse_opt(True)

BATCH_SIZE = opt.batch_size
EPOCHS = opt.epochs

model_type = opt.model_type
data_type = opt.data_type
lossfn_str = opt.loss_fn
#set up training
optimizer = tf.keras.optimizers.Adam()
if lossfn_str =='ce':
  loss_fn = tf.keras.losses.CategoricalCrossentropy()
elif lossfn_str =='nll':
  loss_fn = negative_log_likelihood
else:
  loss_fn = tf.keras.losses.CategoricalCrossentropy()
if model_type == "lstm":
    from model.lstm import LSTM
    from config.lstm import Config
    #set up training
    # load model with configure
    config = Config
    config.normalizer = opt.normalizer
    config.n_classes = opt.num_classes 
    config.timestep = opt.sequence_length
    model_lstm = LSTM(config=config)
    model = model_lstm.build()
    print(model.summary())
if model_type =='cnn':
    #set up training
    # load model with configure
    from model.cnn import CNN
    from config.cnn import Config
    config = Config
    config.n_classes = opt.num_classes 
    config.timestep = opt.sequence_length
    config.kernel_size = opt.kernel_size 
    # config.filters = opt.filters
    config.normalizer = opt.normalizer
    model_cnn = CNN(config=config)
    model = model_cnn.build()
    print(model.summary())
if model_type == 'mlp':
    from model.mlp import MLP
    from config.mlp import Config
    config  = Config
    config.n_classes = opt.num_classes
    config.timestep  = opt.sequence_length
    model_mlp = MLP(config=config)
    model = model_mlp.build()
    print(model.summary())
    
if model_type == 'baseline':
    from model.lstm_baseline import LSTM_baseline
    from config.lstm_baseline import Config
    
    config  = Config
    config.n_classes = opt.num_classes
    config.timestep  = opt.sequence_length
    model_baseline = LSTM_baseline(config=config)
    model = model_baseline.build()
    print(model.summary())
    
    
if model_type == 'adaptive_lstm':
    from model.adaptive_lstm import AdaptiveLSTM
    from config.adaptive_lstm import Config
    config  = Config
    config.n_classes = opt.num_classes
    config.timestep  = opt.sequence_length
    config.filters = opt.filters
    config.kernel_size = opt.kernel_size
    config.normalizer = opt.normalizer
    model_adaptive = AdaptiveLSTM(config=config)
    model = model_adaptive.build()
    print(model.summary())
    
    
    
if model_type == 'lstm_v2':
    from model.lstm_v2 import LSTM_v2
    from config.lstm import Config
    config  = Config
    config.n_classes = opt.num_classes
    config.timestep  = opt.sequence_length
    model_v2 = LSTM_v2(config=config)
    model = model_v2.build()
    print(model.summary())

else: 
    from model.lstm import LSTM
    from config.lstm import Config
    #set up training
    # load model with configure
    config = Config
    config.n_classes = opt.num_classes 
    config.timestep = opt.sequence_length
    config.normalizer = opt.normalizer
    model_lstm = LSTM(config=config)
    model = model_lstm.build()

if opt.check_point is not None:
    model = tf.keras.models.load_model(str(opt.check_point))

#load data
print("<=====> Training progress <=====>")
if opt.scenario == "sample_divide":
    print("Combine all data of 16 person and split with ratio at 0.75-0.25 (12-4) for train/test")
    X_train, X_val, y_train, y_val = load_and_process_data(file_path='./data/dataset/trainset_16.npy', sequence_length=config.timestep, overlap=opt.overlap, valid_ratio=0.2)
else:
    X_train, X_val, y_train, y_val = load_from_folder(folder_path=train_folder, sequence_length=config.timestep, overlap=opt.overlap, valid_ratio=0.2)
print("shape of training data: ", X_train.shape)
print("shape of validation data: ", X_val.shape)




train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_dataset = val_dataset.batch(BATCH_SIZE)

print("===Load data passed====")
#set up training
#set up optimizer  chosen by implement in configure
optimizer = tf.keras.optimizers.Adam()
# training progress
print("Training Model ........")
history, model = train_model(model=model, dataset=train_dataset, loss_fn=loss_fn, optimizer=optimizer,epochs=EPOCHS, val_dataset=val_dataset, batch_size=BATCH_SIZE,  arg=opt)
print("=====Training Done !====")
history["test"]  = dict()
#test progress

if opt.scenario == "sample_divide":
    scenario = opt.scenario
    print("<====== Evaluate on testset =======>")
    X_test, y_test = load_and_process_data(file_path='./data/dataset/testset_16.npy',sequence_length= opt.sequence_length, overlap = opt.overlap, valid_ratio=None)
    ds_test = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    ds_test = ds_test.batch(BATCH_SIZE)
    cl_report, cf_matrix, results = test_model(test_set=ds_test, model=model, loss_fn = loss_fn, batch_size=BATCH_SIZE)
    print("Result on testste: \n", results)
    metrics = ["Loss", "Acc", "Lipschitz Loss", "Lipshitz model", "Time"]
    history["test"]['cl_report'] = cl_report
    history["test"]['cf_matrix'] = cf_matrix      
    for metric, result in zip(metrics, results):
        history["test"][metric] = result


    cols = ["Person", "Loss", "Acc", "L Loss", "L model"]
        
    df_total = dict()
    df_total['Person'] = "Eval All"
    for i in range(4):
        df_total[cols[i+1]] = [results[i]]
      
      
    file_excel_total = "./work_dir/hist_{}_{}_{}_{}_{}_{}_{}/test_total_history_{}_{}_{}_{}_{}_{}.csv".format(model_type, data_type, opt.sequence_length, opt.overlap,scenario,lossfn_str, opt.normalizer, EPOCHS, BATCH_SIZE,  scenario, today, lossfn_str, opt.normalizer)
      
    # file_excel_total = "./work_dir/hist_{}_{}_{}_{}_{}_{}_{}/test_history_eval_total_{}_{}_{}_{}_{}_{}.csv".format(model_type, data_type, opt.sequence_length, opt.overlap,scenario,lossfn_str, opt.normalizer, EPOCHS, BATCH_SIZE,  scenario, today, lossfn_str, opt.normalizer)
    os.makedirs(os.path.dirname(file_excel_total), exist_ok=True)
    df = pd.DataFrame(df_total, columns=cols)
    df.to_csv(file_excel_total)
    print("test_done!!!!!")

  
    history["test_neck"] = dict()
    for file in os.listdir(test_neck_path):  
        name = file.split("_")[-1]
        print(f"<=======>Test on {name[:-4]}'s data<=======>\n")
        file_dir = f"{test_neck_path}/{file}"
        X_test, y_test = load_and_process_data(file_path=file_dir,sequence_length= opt.sequence_length, overlap = opt.overlap, valid_ratio=None)
        ds_test = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        ds_test = ds_test.batch(BATCH_SIZE)
        cl_report, cf_matrix,results_neck = test_model(test_set=ds_test, model=model, loss_fn = loss_fn, batch_size=BATCH_SIZE)
        print(f"Result on {name}'s data(neckdata): \n", results_neck)
        metrics = ["Loss", "Acc", "Lipschitz Loss", "Lipshitz model", "Time"]
        history["test_neck"][name[:-4]] = dict()
        history["test_neck"][name[:-4]]['cl_report'] = cl_report
        history["test_neck"][name[:-4]]['cf_matrix'] = cf_matrix
        # experiment.log_confusion_matrix(
        #         cm=cf_matrix)    
        for metric, result in zip(metrics, results_neck):
            history["test_neck"][name[:-4]][metric] = result
            
        
        
else:
    for file in os.listdir(test_path):  
        scenario = 'person_divide'
        name = file.split("_")[-1]
        print(f"<=======>Test on {name[:-4]}'s data<=======>\n")
        file_dir = f"{test_path}/{file}"
        X_test, y_test = load_and_process_data(file_path=file_dir,sequence_length= opt.sequence_length, overlap = opt.overlap, valid_ratio=None)
        ds_test = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        ds_test = ds_test.batch(BATCH_SIZE)
        cl_report, cf_matrix,results = test_model(test_set=ds_test, model=model, loss_fn = loss_fn, batch_size=BATCH_SIZE)
        print(f"Result on {name}'s data: \n", results)
        metrics = ["Loss", "Acc", "Lipschitz Loss", "Lipshitz model", "Time"]
        history["test"][name[:-4]] = dict()
        history["test"][name[:-4]]['cl_report'] = cl_report
        history["test"][name[:-4]]['cf_matrix'] = cf_matrix   
        for metric, result in zip(metrics, results):
            history["test"][name[:-4]][metric] = result
            
    history["test_neck"] = dict()
    for file in os.listdir(test_neck_path):  
        scenario = 'person_divide'
        name = file.split("_")[-1]
        print(f"<=======>Test on {name[:-4]}'s data<=======>\n")
        file_dir = f"{test_neck_path}/{file}"
        X_test, y_test = load_and_process_data(file_path=file_dir,sequence_length= opt.sequence_length, overlap = opt.overlap, valid_ratio=None)
        ds_test = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        ds_test = ds_test.batch(BATCH_SIZE)
        cl_report, cf_matrix,results = test_model(test_set=ds_test, model=model, loss_fn = loss_fn, batch_size=BATCH_SIZE)
        print(f"Result on {name}'s data(neckdata): \n", results)
        metrics = ["Loss", "Acc", "Lipschitz Loss", "Lipshitz model", "Time"]
        history["test_neck"][name[:-4]] = dict()
        history["test_neck"][name[:-4]]['cl_report'] = cl_report
        history["test_neck"][name[:-4]]['cf_matrix'] = cf_matrix
        # experiment.log_confusion_matrix(
        #         cm=cf_matrix)    
        for metric, result in zip(metrics, results):
            history["test_neck"][name[:-4]][metric] = result
            
            
            
        
        

# checkpoint_pth = f"./checkpoint/checkpoint_{opt.model_type}_{opt.data_type}_{opt.sequence_length}_{opt.overlap}_{scenario}/{model_type}_{data_type}_{opt.sequence_length}_{opt.overlap}_{scenario}.keras"
# os.makedirs(os.path.dirname(checkpoint_pth), exist_ok=True)
# model.save(f"./checkpoint/checkpoint_{opt.model_type}_{opt.data_type}_{opt.sequence_length}_{opt.overlap}_{scenario}/{model_type}_{data_type}_{opt.sequence_length}_{opt.overlap}_{scenario}.keras")
# print(history)
    import numpy as np
    arr  = np.zeros((4, 4))
    persons = []
    for i, (person, metrics) in enumerate(history["test"].items()):
        print(person, metrics)
        arr[i][0]  = metrics["Loss"]
        arr[i][1] = metrics["Acc"]
        arr[i][2] = metrics["Lipschitz Loss"]
        arr[i][3] = metrics["Lipshitz model"]
        persons.append(person)


    mean = np.mean(arr, axis = 0 ).reshape(1, 4)
    std =  np.std(arr, axis = 0).reshape(1,4)
    print("mean_exp", mean)
    print("std_exp", std)
    persons.append("Mean")
    persons.append("Standard Deviation")
    df_np = np.concatenate([arr, mean, std], axis = 0 )
    cols = ["Person", "Loss", "Acc", "L Loss", "L model"]
    df_dict = dict()
    df_dict["Person"] = persons
    for i in range(4):
        df_dict[cols[i+1]] = df_np[:, i]


    # save test history on excel file
    file_excel = "./work_dir/hist_{}_{}_{}_{}_{}_{}_{}/test_history_{}_{}_{}_{}_{}_{}.csv".format(model_type, data_type, opt.sequence_length, opt.overlap,scenario,lossfn_str, opt.normalizer, EPOCHS, BATCH_SIZE,  scenario, today, lossfn_str, opt.normalizer)
    os.makedirs(os.path.dirname(file_excel), exist_ok=True)
    df = pd.DataFrame(df_dict, columns=cols)
    df.to_csv(file_excel)




import numpy as np
arr_neck  = np.zeros((2, 4))
cols = ["Person", "Loss", "Acc", "L Loss", "L model"]
persons_neck = []
for i, (person, metrics) in enumerate(history["test_neck"].items()):
  print(person, metrics)
  arr_neck[i][0]  = metrics["Loss"]
  arr_neck[i][1] = metrics["Acc"]
  arr_neck[i][2] = metrics["Lipschitz Loss"]
  arr_neck[i][3] = metrics["Lipshitz model"]
  persons_neck.append(person)
  
print(arr_neck)
df_neck = dict()
df_neck['Person'] = persons_neck
for i in range(4):
  df_neck[cols[i+1]] = arr_neck[:, i]
  
  
file_excel_neck = "./work_dir/hist_{}_{}_{}_{}_{}_{}_{}/test_neck_history_{}_{}_{}_{}_{}_{}.csv".format(model_type, data_type, opt.sequence_length, opt.overlap,scenario,lossfn_str, opt.normalizer, EPOCHS, BATCH_SIZE,  scenario, today, lossfn_str, opt.normalizer)
os.makedirs(os.path.dirname(file_excel_neck), exist_ok=True)
df_neck_data = pd.DataFrame(df_neck, columns=cols)
df_neck_data.to_csv(file_excel_neck)



filename = "./work_dir/hist_{}_{}_{}_{}_{}_{}_{}/training_history_{}_{}_{}_{}_{}_{}.pkl".format(model_type, data_type, opt.sequence_length, opt.overlap,scenario,lossfn_str, opt.normalizer, EPOCHS, BATCH_SIZE,  scenario, today, lossfn_str, opt.normalizer)
os.makedirs(os.path.dirname(filename), exist_ok=True)
with open(filename, 'wb') as  f:
    pickle.dump(history, f)
    

plot_performance(history=history, model_type=model_type, arg=opt)

if __name__ == "__main__":
    opt = parse_opt(True)


# from config.adaptive_lstm import Config
# from model.adaptive_lstm import AdaptiveLSTM
# model = AdaptiveLSTM(config=Config)
# model = model.build()
# print(model.summary())
