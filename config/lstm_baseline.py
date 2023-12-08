class Config:
    timestep  = 20
    n_features = 3
    n_classes  = 12
    hidden_size = [25]
    mlp_units  = [32]
    dropout = 0.4 
    log_dir = "checkpoint/lstm/"
    save_file =  "checkpoint/lstm_model/"
    activation = 'tanh'
    regularizers = 'l1'
    normalizer = 'batch_norm'
    lr  = 0.01,
    optimizer = 'adam',
    loss_fn = 'categorical-crossentropy'
    
