class Config:
    timestep = 100
    n_features = 3
    head_size = 256 
    num_heads = 4
    filter = 4
    num_encoder_blocks = 6
    mlp_units = [128, 64]
    drop_out = 0.4
    n_classes = 5
    lr = 0.01
    # activation = 'tanh'
    regularizers = None
    normalizer = 'batch_norm'
    optimizer = 'adam'
    loss_fn = 'categorical-crossentropy'
    lr  = 0.01,
    log_dir = "checkpoint/transformer/"
    save_file = "checkpoint/transformer_model/"
    activation = 'tanh'
    
# 