LSTM_config = dict(
    timestep = 1, 
    n_features = 3,
    hidden_size = [128,  64],
    mlp_units = [32],
    drop_out = 0.4,
    n_classes = 5,
    # learning_rate = 1e-4, 
    log_dir = "checkpoint/lstm/",
    save_file = "checkpoint/lstm_model/",
    activation = 'tanh'
)

Transformer_config = dict(
    timestep = 1, 
    n_features = 3,
    head_size = 256, 
    num_heads = 4,
    filter = 4, 
    num_encoder_blocks = 6, 
    mlp_units = [128, 64],
    drop_out = 0.4,
    n_classes = 5,
    log_dir = "checkpoint/transformer/",
    save_file = "checkpoint/transformer_model/",
    activation = 'tanh'
    )


CNN_config = dict(
    timestep = 1, 
    n_features = 3,
    filters = [128, 64],
    mlp_units = [128, 64],
    kernel_size = 3,
    drop_out = 0.4,
    n_classes = 5,
    log_dir = "checkpoint/cnn/",
    save_file = "checkpoint/cnn_model/",
    activation = 'tanh'
)
MLP_config = dict(
    timestep = 1, 
    n_features = 3,
    filters = [128, 64],
    mlp_units = [128, 64],
    kernel_size = 3,
    drop_out = 0.4,
    n_classes = 5,
    log_dir = "checkpoint/mlp/",
    save_file = "checkpoint/mlp_model/",
    activation = 'tanh'
)
Complier= dict(
    lr = 1e-3,
    optimizer = ['adam', 'SGD'],
    loss = ['categorical_crossentropy']
)