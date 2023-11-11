import pickle
import os
import tensorflow as tf
from model.lstm import LSTM
from config.lstm import Config
from utils.trainer import train_model
from utils.load_data import load_and_process_data, tensorflow_dataset, load_from_folder
from utils.trainer import test_model
from utils.parse import parse_opt
from utils.plot import plot_performance
#load data
# train_path = 'data/dynamic1/train/train_PhamQuangTu.npy'

train_folder = 'data/new_data_static/trainset'
test_path = 'data/new_data_static/testset'

opt = parse_opt(True)

BATCH_SIZE = opt.batch_size
EPOCHS = opt.epochs

model_type = opt.model_type
data_type = opt.data_type
if model_type == "lstm":
    from model.lstm import LSTM
    from config.lstm import Config
    #set up training
    # load model with configure
    config = Config
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


else: 
    from model.lstm import LSTM
    from config.lstm import Config
    #set up training
    # load model with configure
    config = Config
    config.n_classes = opt.num_classes 
    config.timestep = opt.sequence_length
    model_lstm = LSTM(config=config)
    model = model_lstm.build()

if opt.check_point is not None:
    # lastest = tf.train.latest_checkpoint(str(opt.check_point))
    model = tf.keras.models.load_model(str(opt.check_point))

#load data

print("<=====> Training progress <=====>")

print("<=====> Training progress <=====>")
if opt.scenario is not None:
    print("Combine all data of 16 person and split with ratio at 0.75-0.25 (12-4) for train/test")
    X_train, X_val, y_train, y_val = load_and_process_data(file_path='./data/dataset/trainset_16.npy', sequence_length=config.timestep, overlap=opt.overlap, valid_ratio=0.2)
else:
    X_train, X_val, y_train, y_val = load_from_folder(folder_path=train_folder, sequence_length=config.timestep, overlap=opt.overlap, valid_ratio=0.2)
print("shape of training data: ", X_train.shape)
print("shape of validation data: ", X_val.shape)
ds_train = tensorflow_dataset(data=X_train, labels=y_train, batch_size=BATCH_SIZE)
ds_val = tensorflow_dataset(data=X_val, labels=y_val, batch_size=BATCH_SIZE)
#set up training
# load model with configure

# print(model.summary())

estimate_interval_loss_fn = 10
estimate_interval_model = 10

loss_fn = tf.keras.losses.CategoricalCrossentropy()
#set up optimizer  chosen by implement in configure
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# training progress
print("Training Model ........")
history, model = train_model(model=model, dataset=ds_train, loss_fn=loss_fn, optimizer=optimizer,epochs=EPOCHS, val_dataset=ds_val, arg=opt)

history["test"]  = dict()
#test progress

if opt.scenario is not None:
    scenario = opt.scenario
    print("<====== Evaluate on testset =======>")

    X_test, y_test = load_and_process_data(file_path='./data/dataset/testset_16.npy',sequence_length= opt.sequence_length, overlap = opt.overlap, valid_ratio=None)
    ds_test = tensorflow_dataset(data=X_test, labels=y_test, batch_size=BATCH_SIZE)
    results = test_model(dataset=ds_test, model=model, loss_fn = loss_fn)
    print("Result on testste: \n", results)
    metrics = ["Loss", "Acc", "Lipschitz Loss", "Lipshitz model"]
    history["test"] = dict()
    for metric, result in zip(metrics, results):
        history["test"][metric] = result
else:
    for file in os.listdir(test_path):  
        scenario = 'person_divide'
        name = file.split("_")[-1]
        print(f"<=======>Test on {name[:-4]}'s data<=======>\n")
        file_dir = f"{test_path}/{file}"
        X_test, y_test = load_and_process_data(file_path=file_dir,sequence_length= opt.sequence_length, overlap = opt.overlap, valid_ratio=None)
        ds_test = tensorflow_dataset(data=X_test, labels=y_test, batch_size=BATCH_SIZE)
        results = test_model(dataset=ds_test, model=model, loss_fn = loss_fn)
        print(f"Result on {name}'s data: ", results)
        metrics = ["Loss", "Acc", "Lipschitz Loss", "Lipshitz model"]
        history["test"][name[:-4]] = dict()
        for metric, result in zip(metrics, results):
            history["test"][name[:-4]][metric] = result
checkpoint_pth = f"./checkpoint/checkpoint_{opt.model_type}_{opt.data_type}_{opt.sequence_length}_{opt.overlap}_{scenario}/{model_type}_{data_type}_{opt.sequence_length}_{opt.overlap}_{scenario}.keras"
os.makedirs(os.path.dirname(checkpoint_pth), exist_ok=True)
model.save(f"./checkpoint/checkpoint_{opt.model_type}_{opt.data_type}_{opt.sequence_length}_{opt.overlap}_{scenario}/{model_type}_{data_type}_{opt.sequence_length}_{opt.overlap}_{scenario}.keras")
print(history)
filename = "./work_dir/hist_{}_{}_{}_{}_{}/training_history_{}_{}_{}.pkl".format(model_type, data_type, opt.sequence_length, opt.overlap,scenario, EPOCHS, BATCH_SIZE,  scenario)
os.makedirs(os.path.dirname(filename), exist_ok=True)
with open("./work_dir/hist_{}_{}_{}_{}_{}/training_history_{}_{}_{}.pkl".format(model_type, data_type, opt.sequence_length, opt.overlap,scenario, EPOCHS, BATCH_SIZE,  scenario), 'wb') as  f:
    pickle.dump(history, f)
plot_performance(history=history, model_type=model_type, arg=opt)

if __name__ == "__main__":
    opt = parse_opt(True)
