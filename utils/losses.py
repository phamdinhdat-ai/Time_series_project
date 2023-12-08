import tensorflow as tf

def negative_log_likelihood(predictions, targets):
    epsilon = 1e-15  # Small value to prevent log(0)
    # Calculate negative log likelihood
    nll = -tf.reduce_mean(tf.reduce_sum(targets * tf.math.log(predictions + epsilon), axis=1))
    return nll
