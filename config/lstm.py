
class Config:
    timestep  = 20
    n_features = 3
    n_classes  = 13
    hidden_size = [8, 16]
    mlp_units  = [16]
    dropout = 0.4 
    log_dir = "checkpoint/lstm/"
    save_file =  "checkpoint/lstm_model/"
    activation = 'tanh'
    regularizers = None
    normalizer = 'batch_norm'
    lr  = 0.01,
    optimizer = 'adam',
    loss_fn = 'categorical-crossentropy'
    
