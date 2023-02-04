
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns


def plot_performence(model_history,epochs= 50,  model_name= "LSTM", validation = True):
    if validation==True:
        # plot accuracy
        plt.plot(model_history.history['acc']) #variable 1 to plot
        plt.plot(model_history.history['val_acc']) #variable 2 to plot
        plt.title(f'Model {model_name} accuracy') #title
        plt.ylabel('Accuracy') #label y
        plt.xlabel('Epoch') #label x
        plt.legend(['Training', 'Validation'], loc='lower right') #legend
        plt.show()
        # plt.savefig(f"acc_{len(epochs)}.png")
        # plot losss 
        plt.plot(model_history.history['loss'])
        plt.plot(model_history.history['val_loss'])
        plt.title(f'Model {model_name} loss') #title
        plt.ylabel('Loss') #label y
        plt.xlabel('Epoch') #label x
        plt.legend(['Training', 'Validation'], loc='upper right') #legend
        plt.show()

    else: 
        #plot accuracy
        plt.plot(model_history.history['acc']) #variable 1 to plot
        plt.title(f'Model {model_name} accuracy') #title
        plt.ylabel('Accuracy') #label y
        plt.xlabel('Epoch') #label x
        plt.legend(['Training'], loc='lower right') #legend
        plt.show()
        #plot loss
        plt.plot(model_history.history['loss']) #variable 1 to plot
        plt.title(f'Model {model_name} loss') #title
        plt.ylabel('Loss') #label y
        plt.xlabel('Epoch') #label x
        plt.legend(['Training'], loc='upper right') #legend
        plt.show()
        
def plot(y_pred, y_true, label_names:list,  save_file:str):
    
    from sklearn.metrics import confusion_matrix
    cf_matrix = confusion_matrix(y_pred=y_pred, y_true=y_true)
    import seaborn as sns

    ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')

    ax.set_title('Confusion Matrix\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(label_names)
    ax.yaxis.set_ticklabels(label_names)

    ## Display the visualization of the Confusion Matrix.
    plt.show()


