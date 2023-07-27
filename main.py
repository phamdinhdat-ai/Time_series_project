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
train_path = 'data/dynamic1/train/train_PhamQuangTu.npy'
test_path = 'data/dynamic2/test'

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

else: 
    from model.lstm import LSTM
    from config.lstm import Config
    #set up training
    # load model with configure
    config = Config
    config.n_classes = opt.num_classes 
    config.timestep = opt.sequence_length
    model_lstm = LSTM(config=config)



#load data
train_folder = 'data/dynamic2/train'
print("<=====> Training progress <=====>")
X_train, X_val, y_train, y_val = load_from_folder(folder_path=train_folder, sequence_length=config.timestep, overlap=opt.overlap, valid_ratio=0.2)
print("shape of training data: ", X_train.shape)
print("shape of validation data: ", X_val.shape)
ds_train = tensorflow_dataset(data=X_train, labels=y_train, batch_size=BATCH_SIZE)
ds_val = tensorflow_dataset(data=X_val, labels=y_val, batch_size=BATCH_SIZE)
#set up training
# load model with configure

print(model.summary())

estimate_interval_loss_fn = 10
estimate_interval_model = 10

loss_fn = tf.keras.losses.CategoricalCrossentropy()
#set up optimizer  chosen by implement in configure
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

#set up loss function chosen by implement in configure
# if Config.loss_fn == 'categorical-crossentropy':
#     loss_fn = tf.keras.losses.CategoricalCrossentropy()
# if Config.loss_fn == 'sparse-categorical-crossentropy':
#     loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
# if Config.loss_fn == 'binary-crossentropy':
#     loss_fn = tf.keras.losses.BinaryCrossentropy()
# if Config.loss_fn == 'mse':
#     loss_fn = tf.keras.losses.MeanSquaredError()
# if Config.loss_fn == 'mae':
#     loss_fn = tf.keras.losses.MeanAbsoluteError()
# else: 
#     loss_fn = tf.keras.losses.CategoricalCrossentropy()

# #set up optimizer  chosen by implement in configure

# if Config.optimizer == "adam":
#     optimizer = tf.keras.optimizers.Adam(learning_rate=Config.lr)
# if Config.optimizer == "adamax":
#     optimizer = tf.keras.optimizers.Adamax(learning_rate=Config.lr)
# if Config.optimizer == "SGD":
#     optimizer = tf.keras.optimizers.SGD(learning_rate=Config.lr)
# else:
#     optimizer = tf.keras.optimizers.Adam(learning_rate=Config.lr)

# training progress
history, model = train_model(model=model, dataset=ds_train, loss_fn=loss_fn, optimizer=optimizer,epochs=EPOCHS, val_dataset=ds_val)

history["test"]  = dict()
#test progress

for file in os.listdir(test_path):  
    _, name = file.split("_")
    print(f"<=======>Test on {name[:-4]}'s data<=======>\n")
    file_dir = f"{test_path}/{file}"
    X_test, y_test = load_and_process_data(file_path=file_dir,valid_ratio=None)
    ds_test = tensorflow_dataset(data=X_test, labels=y_test, batch_size=BATCH_SIZE)
    results = test_model(dataset=ds_test, model=model, loss_fn = loss_fn)
    print(f"Result on {name}'s data: ", results)
    metrics = ["Loss", "Acc", "Lipschitz Loss", "Lipshitz model"]
    history["test"][name[:-4]] = dict()
    for metric, result in zip(metrics, results):
        history["test"][name[:-4]][metric] = result

print(history)
with open("{}_{}_{}_{}_{}_{}.pkl".format(model_type,data_type,EPOCHS, BATCH_SIZE, opt.sequence_length, opt.overlap), 'wb') as  f:
    pickle.dump(history, f)
plot_performance(history=history, model_type=model_type)

if __name__ == "__main__":
    opt = parse_opt(True)