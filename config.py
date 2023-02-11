LSTM_config = dict(
    timestep = 1, 
    n_features = 3,
    hidden_size = [256, 128],
    mlp_units = [64],
    drop_out = 0.4,
    n_classes = 5,
    # learning_rate = 1e-4, 
    log_dir = "/checkpoint/LSTM/",
    save_file = "/content/drive/MyDrive/Checkpoint/",
    activation = 'tanh'
)
sequence_lenght = 20,
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
    log_dir = "./checkpoint/Transformer/",
    save_file = r"/content/drive/MyDrive/Checkpoint",
    activation = 'tanh'
    )


CNN_config = dict(
    timestep = 10, 
    n_features = 3,
    filters = [128, 64],
    mlp_units = [128, 64],
    drop_out = 0.4,
    n_classes = 5,
    log_dir = "/checkpoint/cnn/",
    save_file = "/checkpoint/work_dir_cnn/",
    activation = 'tanh'
)
Compling = dict(
    lr = [1e-3],
    optimizer = ['adam', 'SGD'],
    loss = ['categorical_crossentropy']
)