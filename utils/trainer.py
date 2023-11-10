import tensorflow as tf 
from config.base_config import BaseConfig
from config.lstm import Config
from model.lstm import LSTM
from utils.lipschitz import *
import keras
import numpy as np 
import pickle
import time
# Prepare the metrics.
acc_metric = keras.metrics.Accuracy()

estimate_interval_loss_fn = 10
estimate_interval_model = 10

#set up loss function chosen by implement in configure
if BaseConfig.loss_fn == 'categorical-crossentropy':
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
if BaseConfig.loss_fn == 'sparse-categorical-crossentropy':
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
if BaseConfig.loss_fn == 'binary-crossentropy':
    loss_fn = tf.keras.losses.BinaryCrossentropy()
if BaseConfig.loss_fn == 'mse':
    loss_fn = tf.keras.losses.MeanSquaredError()
if BaseConfig.loss_fn == 'mae':
    loss_fn = tf.keras.losses.MeanAbsoluteError()

#set up optimizer  chosen by implement in configure

if BaseConfig.optimizer == "adam":
    optimizer = tf.keras.optimizers.Adam(learning_rate=Config.lr)
if BaseConfig.optimizer == "adamax":
    optimizer = tf.keras.optimizers.Adamax(learning_rate=Config.lr)
if BaseConfig.optimizer == "SGD":
    optimizer = tf.keras.optimizers.SGD(learning_rate=Config.lr)


# # model 

# base_model = LSTM(config=Config)
# model = base_model.build()

def accuracy_score(y_true, y_pre):
    count  = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pre[i]:
            count +=1 
    return count/len(y_true)

import keras

train_acc_metric = keras.metrics.Accuracy()
val_acc_metric = keras.metrics.Accuracy()

def train_model(model,
                dataset,
                loss_fn,
                optimizer,
                epochs= 100,  
                estimate_interval = 10,
                val_dataset=None):
    loss_model = []
    acc_model  = []
    lipschitz_loss = []
    lipschitz_model = []
    val_loss_model = []
    val_acc_model  = []
    val_lipschitz_loss = []
    val_lipschitz_model = []
    time_run_train = []
    time_run_val = []
    # history = dict()
    history = dict(
    Loss = [],
    Acc = [],
    L_loss = [],
    L_model = [],
    Time_train = [],
    Val_loss = [],
    Val_acc = [],
    Val_L_loss = [],
    Val_L_model = [],
    Time_val = [],
    )
    for epoch in range(epochs):
        eloss = []
        eacc  = []
        elipschitz_loss = []
        elipschitz_model = []
        val_eloss = []
        val_eacc  = []
        val_elipschitz_loss = []
        val_elipschitz_model = []
        time_e = 0
        mean_acc = 0
        mean_loss = 0 
        mean_l_loss = 0
        mean_l_model = 0
        mean_val_acc = 0
        mean_val_loss = 0 
        mean_val_l_loss = 0
        mean_val_l_model = 0
        time_train = 0
        time_val = 0
        start = time.time()
        for step,(inputs, labels) in enumerate(dataset):
            
            with tf.GradientTape() as tape:
                # forward pass 
                output = model(inputs, training=True)
                #compute loss
                loss = loss_fn(labels, output)
            #compute gradients over parameters
            gradients = tape.gradient(loss, model.trainable_variables)
            #update model's parameters 
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            predicted_labels = tf.argmax(output, axis=1)
            y_true = tf.argmax(labels, axis = 1)
            # Update training metric.
            train_acc_metric.update_state(y_true, predicted_labels)
            # Display metrics at the end of each epoch.
            train_acc = train_acc_metric.result()
            # Reset training metrics at the end of each epoch
            train_acc_metric.reset_states()
            # Estimate the Lipschitz constant of the loss function periodically
            
            if step % estimate_interval == 0:
                lipschitz_constant_loss_fn = estimate_lipschitz_constant_loss_fn(model, loss_fn, inputs, labels)
                # print("\n")
                # tf.print("Estimated Lipschitz constant of the loss function:", lipschitz_constant_loss_fn)

            # Estimate the Lipschitz constant of the model periodically
            if step % estimate_interval == 0:
                lipschitz_constant_model = estimate_lipschitz_constant_model_v1(model,inputs)
                # print("\n")
                # tf.print("Estimated Lipschitz constant of the model:", lipschitz_constant_model)
            time_train_batch = time.time() - start
            eloss.append(loss)
            eacc.append(train_acc)
            elipschitz_loss.append(lipschitz_constant_loss_fn)
            elipschitz_model.append(lipschitz_constant_model)
            mean_acc += train_acc 
            mean_loss +=loss 
            mean_l_loss +=lipschitz_constant_loss_fn
            mean_l_model += lipschitz_constant_model
            time_train +=time_train_batch

        start2 = time.time()
        #compute validaion results 
        for x_batch_val, y_batch_val in val_dataset:
            val_out = model(x_batch_val, training=False)
            #compute val loss
            val_loss = loss_fn(val_out, y_batch_val)

            #compute lipschitz constant loss function on valid set 
            val_l_loss = estimate_lipschitz_constant_loss_fn(model, loss_fn, x_batch_val, y_batch_val)
            #compute lipschittz constant model on valid set
            val_l_model = estimate_lipschitz_constant_model_v1(model, x_batch_val)
            # Update val metrics
            predicted_labels_val = tf.argmax(val_out, axis=1)
            y_true_val = tf.argmax(y_batch_val, axis = 1)
            val_acc_metric.update_state(y_true_val, predicted_labels_val)
            val_acc = val_acc_metric.result()
            val_acc_metric.reset_states()
            time_val_batch = time.time() - start2

            val_eloss.append(val_loss)
            val_eacc.append(val_acc)
            val_elipschitz_loss.append(val_l_loss)
            val_elipschitz_model.append(val_l_model)
            # accuracy, loss = train_step(inputs, labels)
            mean_val_acc +=val_acc
            mean_val_loss +=val_loss
            mean_val_l_loss  += val_l_loss 
            mean_val_l_model += val_l_model
            time_val += time_val_batch

            # eloss.append(loss)
            # eacc.append(train_acc)
            # elipschitz_loss.append(lipschitz_constant_loss_fn)
            # elipschitz_model.append(lipschitz_constant_model)
            # #story validation result per epoch
            # # val_eloss.append(val_loss)
            # # val_eacc.append(val_acc)
            # # val_elipschitz_loss.append(val_l_loss)
            # # val_elipschitz_model.append(val_l_model)
            # # accuracy, loss = train_step(inputs, labels)
            # mean_acc += train_acc 
            # mean_loss +=loss 
            # mean_l_loss +=lipschitz_constant_loss_fn
            # mean_l_model += lipschitz_constant_model
            # mean_val_acc +=val_acc
            # mean_val_loss +=val_loss
            # mean_val_l_loss  += val_l_loss 
            # mean_val_l_model += val_l_model

        print(f"\rEpoch: {epoch}| Loss: {mean_loss/(len(dataset))} | Acc: {mean_acc/(len(dataset))} | Lipschitz Loss: {mean_l_loss/(len(dataset))}  | Lipschitz Model: {mean_l_model/(len(dataset))}|Time training: {time_train}| Val Loss: {mean_val_loss/(len(val_dataset))}| Val_Acc: {mean_val_acc/len(val_dataset)}| Val L_loss: {mean_val_l_loss/len(val_dataset)}| Val L_model: {mean_val_l_model/len(val_dataset)}| Time Val: {time_val}" , end=" ", flush=True)

            # print(f"\rEpoch: {epoch}| Steps:{'-'*step}  | Loss: {loss} | Acc: {acc} | Lipschitz Loss: {lipschitz_constant_loss_fn}  | Lipschitz Model: {lipschitz_constant_model}" , end=" ", flush=True)
        print("\n")
        loss_model.append(eloss)
        acc_model.append(eacc)
        lipschitz_loss.append(elipschitz_loss)
        lipschitz_model.append(elipschitz_model)
        time_run_train.append(time_train)
        val_loss_model.append(val_eloss)
        val_acc_model.append(val_eacc)
        val_lipschitz_loss.append(val_elipschitz_loss)
        val_lipschitz_model.append(val_elipschitz_model)
        time_run_val.append(time_val)
        # history["Loss"] = np.array(loss_model).mean(axis=1)
        # history["Acc"] = np.array(acc_model).mean(axis=1)
        # history["L_loss"] = np.array(lipschitz_loss).mean(axis=1)
        # history["L_model"] = np.array(lipschitz_model).mean(axis=1)
        # history['Time_train'] = np.array(time_run_train)
        # history["Val_loss"] = np.array(val_loss_model).mean(axis=1)
        # history["Val_Acc"] = np.array(val_acc_model).mean(axis=1)
        # history["Val_L_loss"] = np.array(val_lipschitz_loss).mean(axis=1)
        # history["Val_L_model"] = np.array(val_lipschitz_model).mean(axis=1)
        # history['Time_val'] = np.array(time_run_val)
        history["Loss"].append(mean_loss/(len(dataset)))
        history["Acc"].append(mean_acc/len(dataset)))
        history["L_loss"].append(mean_l_loss/len(dataset))
        history["L_model"].append(mean_l_model/len(dataset))
        history['Time_train'].append(time_train)
        history["Val_loss"].append(mean_val_loss/len(val_dataset))
        history["Val_Acc"].append(mean_val_acc/len(val_dataset))
        history["Val_L_loss"].append(mean_val_l_loss/len(val_dataset))
        history["Val_L_model"].append(mean_val_l_model/len(val_dataset))
        history['Time_val'].append(time_val)
        if epoch%10 == 0:
            model.save(f"./checkpoint/{model.__class__.__name__}_{epoch}.keras")
            with open("./work_dir/training_history_{}_{}.pkl".format(model.__class__.__name__,epoch), 'wb') as  f:
                pickle.dump(history, f)
    return history, model
def test_model(dataset, model, loss_fn):
    t_loss = []
    t_acc  = []
    t_l_loss = []
    t_l_model = []
    for x_batch, y_batch in dataset:
        out = model(x_batch, training=False)
        #compute val loss
        loss = loss_fn(out, y_batch)

        #compute lipschitz constant loss function on valid set 
        l_loss = estimate_lipschitz_constant_loss_fn(model, loss_fn, x_batch, y_batch)
        #compute lipschittz constant model on valid set
        l_model = estimate_lipschitz_constant_model_v1(model, x_batch)
        # Update val metrics
        predicted_labels = tf.argmax(out, axis=1)
        y_true = tf.argmax(y_batch, axis = 1)
        acc_metric.update_state(y_true, predicted_labels)
        acc = acc_metric.result()
        acc_metric.reset_states()

        t_loss.append(loss)
        t_acc.append(acc)
        t_l_loss.append(l_loss)
        t_l_model.append(l_model)
    
    concated = np.concatenate([t_loss, t_acc, t_l_loss, t_l_model]).reshape(4, len(t_loss))
    return concated.mean(axis = 1)

