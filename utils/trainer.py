

from comet_ml import Experiment
experiment = Experiment(
  api_key="7lyVL5fNdeeqtYZK9Smpz5RGX",
  project_name="journal-2023",
  workspace="datphamai"
)
import tensorflow as tf 
from config.base_config import BaseConfig
from config.lstm import Config
from model.lstm import LSTM
from utils.lipschitz import *
import keras
import numpy as np 
import pickle
import time
import os 
import time
from datetime import date
import tensorboard

today = str(date.today())
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




# Prepare the metrics.
acc_metric = keras.metrics.Accuracy()
train_acc_metric = keras.metrics.Accuracy()
val_acc_metric = keras.metrics.Accuracy()




def train_model(model,
                dataset,
                loss_fn,
                optimizer,
                epochs= 100,  
                batch_size = 512,
                val_dataset=None,
                arg = None):
    
    history = dict(
        Loss = [],
        Acc = [],
        L_loss = [],
        L_model = [],
        Val_loss = [],
        Val_acc = [],
        Val_L_loss = [],
        Val_L_model = [],
        Time = [],
        )
    
    min_val_loss = np.inf
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        start_time = time.time()
        total_train = 0
        loss_e_train = 0
        loss_e_val = 0
        l_train_fn = 0
        l_train_model = 0
        l_val_fn = 0
        l_val_model = 0
        l_t_step = 0
        l_v_step = 0
        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(dataset):
            with tf.GradientTape() as tape:
                logits = model(x_batch_train, training=True)
                loss_value = loss_fn(y_batch_train, logits)
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            loss_e_train += loss_value
            total_train +=1
            predicted_labels = tf.argmax(logits, axis=1)
            y_true = tf.argmax(y_batch_train, axis = 1)
            # Update training metric.
            train_acc_metric.update_state(y_true, predicted_labels)

            # Log every 200 batches.
            if step % 10== 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
                lipschitz_constant_loss_fn = estimate_lipschitz_constant_loss_fn(model, loss_fn, x_batch_train, y_batch_train)
                lipschitz_constant_model = estimate_lipschitz_constant_model_v1(model,x_batch_train)
                l_train_fn += lipschitz_constant_loss_fn
                l_train_model += lipschitz_constant_model
                l_t_step += 1
                print(
                    "Lipchitz constant loss (for one batch) at step %d: %.4f"
                    % (step, float(lipschitz_constant_loss_fn))
                )
                print(
                    "Lipchitz constant Model (for one batch) at step %d: %.4f"
                    % (step, float(lipschitz_constant_model))
                )
                print("Seen so far: %d samples" % ((step + 1) * batch_size))

        # Display metrics at the end of each epoch.
        train_acc = train_acc_metric.result()
        print("Training acc over epoch: %.4f" % (float(train_acc),))
        print("Training loss: %.4f" % (float(loss_e_train/total_train),))
        print("Training Lipchitz_loss_fn: %.4f" % (float(l_train_fn/l_t_step),))
        print("Training Lipchitz_model: %.4f" % (float(l_train_model/l_t_step),))
        #log experiment runtime.
        experiment.log_metric("Accuracy", float(train_acc), epoch=epoch)
        experiment.log_metric("Loss", float(loss_e_train/total_train), epoch=epoch)
        experiment.log_metric("L_Loss", float(l_train_fn/l_t_step), epoch=epoch)
        experiment.log_metric("L_model", float(l_train_model/l_t_step), epoch=epoch)
        
        
        history["Loss"].append(float(loss_e_train/total_train))
        history["Acc"].append(train_acc)
        history["L_loss"].append(float(l_train_fn/l_t_step))
        history["L_model"].append(float(l_train_model/l_t_step))
        print("<============EVAL-TEST==============>")
        # Reset training metrics at the end of each epoch
        train_acc_metric.reset_states()
        total_val = 0
        # Run a validation loop at the end of each epoch.
        for step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
            val_logits = model(x_batch_val, training=False)
            loss_val = loss_fn(y_batch_val, val_logits)
            if step % 10== 0:
                print(
                    "Valid loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_val))
                )
                lipschitz_constant_loss_fn_val = estimate_lipschitz_constant_loss_fn(model, loss_fn, x_batch_val, y_batch_val)
                lipschitz_constant_model_val = estimate_lipschitz_constant_model_v1(model,x_batch_val)
                print(
                    "Lipchitz constant loss on validation set  (for one batch) at step %d: %.4f"
                    % (step, float(lipschitz_constant_loss_fn_val))
                )
                print(
                    "Lipchitz constant Model on validation set  (for one batch) at step %d: %.4f"
                    % (step, float(lipschitz_constant_model_val))
                )
                l_val_fn += lipschitz_constant_loss_fn_val
                l_val_model += lipschitz_constant_model_val
                l_v_step += 1
                print("Seen so far: %d samples" % ((step + 1) * batch_size))

            loss_e_val += loss_val
            total_val +=1
            # Update val metrics
            predicted_labels = tf.argmax(val_logits, axis=1)
            y_true = tf.argmax(y_batch_val, axis = 1)
            val_acc_metric.update_state(y_true, predicted_labels)
        val_acc = val_acc_metric.result()
        val_acc_metric.reset_states()
        print("Validation acc: %.4f" % (float(val_acc),))
        print("Validation loss: %.4f" % (float(loss_e_val/total_val),))
        print("Validation Lipchitz_loss_fn: %.4f" % (float(l_val_fn/l_v_step),))
        print("Validation Lipchitz_model: %.4f" % (float(l_val_model/l_v_step),))
        #log experiment runtime.
        experiment.log_metric("Val Accuracy", float(val_acc), epoch=epoch)
        experiment.log_metric("Val Loss", float(loss_e_val/total_val), epoch=epoch)
        experiment.log_metric("Val L_Loss", float(l_val_fn/l_v_step), epoch=epoch)
        experiment.log_metric("Val L_model", float(l_val_model/l_v_step), epoch=epoch, )
        time_taken = time.time() - start_time
        print("Time taken: %.2fs" % (time.time() - start_time))
        print("=======================================================")

        history["Val_loss"].append(float(loss_e_val/total_val))
        history["Val_acc"].append(val_acc)
        history["Val_L_loss"].append(float(l_val_fn/l_v_step))
        history["Val_L_model"].append(float(l_val_model/l_v_step))
        history['Time'].append(time_taken)
        if min_val_loss > float(loss_e_val/total_val):
            bets_val = float(loss_e_val/total_val)
            best_weights =  f"./checkpoint/checkpoint_{arg.model_type}_{arg.data_type}_{arg.sequence_length}_{arg.overlap}_{ arg.sequence_length}_{arg.loss_fn}_{arg.normalizer}/{arg.model_type}_{today}_best.keras"
            os.makedirs(os.path.dirname(best_weights), exist_ok=True)
            model.save(best_weights)
            min_val_loss = float(loss_e_val/total_val)

        # if epoch%10 == 0:
        #     filename = "./work_dir/hist_{}_{}_{}_{}_{}/training_history_{}_{}_{}_{}_{}.pkl".format(arg.model_type, arg.data_type, arg.sequence_length, arg.overlap,arg.scenario, arg.model_type,epoch, today, arg.loss_fn, arg.normalizer)
        #     checkpoint_pth = f"./checkpoint/checkpoint_{arg.model_type}_{arg.data_type}_{arg.sequence_length}_{arg.overlap}_{ arg.sequence_length}_{arg.loss_fn}_{arg.normalizer}/{arg.model_type}_{epoch}.keras"
        #     os.makedirs(os.path.dirname(checkpoint_pth), exist_ok=True)
        #     os.makedirs(os.path.dirname(filename), exist_ok=True)
        #     model.save(checkpoint_pth)
        #     with open("./work_dir/hist_{}_{}_{}_{}_{}/last_training_history_{}_{}_{}_{}_{}.pkl".format(arg.model_type, arg.data_type, arg.sequence_length, arg.overlap,arg.scenario, arg.model_type,epoch, today, arg.loss_fn, arg.normalizer), 'wb') as  f:
        #         pickle.dump(history, f)
                
    return history, model



from sklearn.metrics import classification_report, confusion_matrix

def test_model(test_set, model, loss_fn,batch_size=512):
    acc_metric.reset_states()
    loss_e_test = 0
    l_test_fn = 0
    l_test_model = 0
    total_test = 0
    l_v_step  = 0
    results = []
    start_time = time.time()
    y_true_total = []
    y_pred_total = []
    # Run a validation loop at the end of each epoch.
    for step, (x_batch, y_batch) in enumerate(test_set):
        test_logits = model(x_batch, training=False)
        loss_test = loss_fn(y_batch, test_logits)
        if step % 1== 0:
            print(
                "Test loss (for one batch) at step %d: %.4f"
                % (step, float(loss_test))
            )
            lipschitz_constant_loss_fn_test = estimate_lipschitz_constant_loss_fn(model, loss_fn, x_batch, y_batch)
            lipschitz_constant_model_test = estimate_lipschitz_constant_model_v1(model,x_batch)
            print(
                "Lipchitz constant loss on Test set  (for one batch) at step %d: %.4f"
                % (step, float(lipschitz_constant_loss_fn_test))
            )
            print(
                "Lipchitz constant Model on Test set  (for one batch) at step %d: %.4f"
                % (step, float(lipschitz_constant_model_test))
            )
            l_test_fn += lipschitz_constant_loss_fn_test
            l_test_model += lipschitz_constant_model_test
            l_v_step += 1
            print("Seen so far: %d samples" % ((step + 1) * batch_size))

        loss_e_test += loss_test
        total_test +=1
        # Update val metrics
        predicted_labels = tf.argmax(test_logits, axis=1)
        y_true = tf.argmax(y_batch, axis = 1)
        y_pred_total.append(predicted_labels)
        y_true_total.append(y_true)
        acc_metric.update_state(y_true, predicted_labels)
    acc = acc_metric.result()
    acc_metric.reset_states()
    print("=====================RESULT===============================")
    print("Test acc: %.4f" % (float(acc),))
    print("Test loss: %.4f" % (float(loss_e_test/total_test),))
    print("Test Lipchitz_loss_fn: %.4f" % (float(l_test_fn/l_v_step),))
    print("Test Lipchitz_model: %.4f" % (float(l_test_model/l_v_step),))
    time_taken = time.time() - start_time
    print("Time taken: %.2fs" % (time.time() - start_time))
    print("=======================================================")
    
    results.append((float(loss_e_test/total_test)))
    results.append((float(acc)))
    results.append((float(l_test_fn/l_v_step)))
    results.append((float(l_test_model/l_v_step)))
    results.append(time_taken)
    
    cl_report = classification_report(np.concatenate(y_pred_total, axis=0), np.concatenate(y_true_total, axis = 0))
    cf_matrix = confusion_matrix(np.concatenate(y_pred_total, axis=0), np.concatenate(y_true_total, axis = 0))
    return  cl_report, cf_matrix, results
    # print(f"Result on {name}'s data: ", results)
    # metrics = ["Loss", "Acc", "Lipschitz Loss", "Lipshitz model", "Time"]
    # history["test"][name[:-4]] = dict()
    # for metric, result in zip(metrics, results):
    #     history["test"][name[:-4]][metric] = result
