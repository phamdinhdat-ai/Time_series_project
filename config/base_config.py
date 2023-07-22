class BaseConfig:
    epochs = 10
    step_per_epochs = 5
    batch_size = 32
    activation = 'tanh'
    regularizers = 'l1'
    normalizer = 'batch norm'
    lr = 0.001,
    optimizer = 'adam',
    loss_fn = 'categorical-crossentropy'
    