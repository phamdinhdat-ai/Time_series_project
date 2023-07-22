
class Config:
    timestep  = 20
    n_features = 3
    n_classes  = 13
    hidden_size = [128, 64]
    mlp_units  = [32]
    dropout = 0.4 
    log_dir = "checkpoint/lstm/"
    save_file =  "checkpoint/lstm_model/"
    activation = 'tanh'
    regularizers = 'l1'
    normalizer = 'batch norm'
    lr  = 0.001,
    optimizer = 'adam',
    loss_fn = 'categorical-crossentropy'
    