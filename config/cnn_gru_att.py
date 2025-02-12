class Config:
    timestep  = 20
    n_features = 3
    n_classes  = 12
    hidden_size = [16]
    mlp_units  = [32]
    dropout = 0.4 
    filters = 8
    kernel_size = 3 
    strides = 1 
    log_dir = "checkpoint/cnn_gru_att/"
    save_file =  "checkpoint/cnn_gru_att/"
    activation = 'tanh'
    regularizers = 'l1'
    normalizer = 'batch_norm'
    lr  = 0.01,
    optimizer = 'adam',
    loss_fn = 'categorical-crossentropy'
    
