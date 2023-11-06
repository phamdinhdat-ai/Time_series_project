import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd 







def plot_performance(history, model_type):
    loss = history['Loss']
    acc = history['Acc']
    l_loss = history['L_loss']
    l_model = history['L_model']
    val_loss = history["Val_loss"] 
    val_acc = history['Val_Acc']
    val_l_loss = history['Val_L_loss']
    val_l_model = history['Val_L_model']


    train_val_loss = np.concatenate([loss, val_loss], axis = 0).reshape(len(loss), 2)
    train_val_acc = np.concatenate([acc, val_acc], axis = 0).reshape(len(acc),2)
    train_val_l_loss = np.concatenate([l_loss, val_l_loss], axis = 0).reshape(len(l_loss),2)
    train_val_l_model = np.concatenate([l_model, val_l_model], axis = 0).reshape(len(l_model),2)


    #plot loss model
    plt.plot(train_val_loss)
    plt.title("Loss Model")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(["Train", "Valid"])
    plt.savefig(f"./work_dir/{model_type}_loss.png")
    plt.show()
    #plot accuracy model 
    plt.plot(train_val_acc)
    plt.title("Accuracy Model")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(["Train", "Valid"])
    plt.savefig(f"./work_dir/{model_type}_acc.png")
    plt.show()
    #plot lipschitz loss function 
    plt.plot(train_val_l_loss)
    plt.title("Lipschitz Loss Function")
    plt.xlabel("Epochs")
    plt.ylabel("Lipschitz contant")
    plt.legend(["Train", "Valid"])
    plt.savefig(f"./work_dir/{model_type}_L_loss.png")
    plt.show()
    #plot lipshitz model 
    plt.plot(train_val_l_model)
    plt.title("Lipschitz Model")
    plt.xlabel("Epochs")
    plt.ylabel("Lipschitz constant")
    plt.legend(["Train", "Valid"])
    plt.savefig(f"./work_dir/{model_type}_L_model.png")
    plt.show()

