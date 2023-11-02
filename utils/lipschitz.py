import tensorflow as tf 


# Define a function to compute the Lipschitz constant of the loss function
@tf.function
def estimate_lipschitz_constant_loss_fn(model, loss_fn, inputs, labels):
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        loss = loss_fn(labels, model(inputs))
    
    gradients = tape.gradient(loss, inputs)
    gradients_norm = tf.norm(gradients)
    
    return gradients_norm

# Define a function to compute the Lipschitz constant of the model
@tf.function
def estimate_lipschitz_constant_model(model, inputs):
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        outputs = model(inputs)
    
    gradients = tape.gradient(outputs, inputs)
    gradients_norm = tf.norm(gradients, axis=1)
    L_model = tf.reduce_max(gradients_norm)
    
    return L_model




import numpy as np
@tf.function
def estimate_lipschitz_constant_model_v1(model, inputs, epsilon=1e-3, max_iters=10):
    # Generate random inputs within the specified shape
    x = inputs
    input_shape = inputs.shape
    # Initialize Lipschitz constant and iterate
    lipschitz_constant = 0
    for _ in range(max_iters):
        # Generate random perturbation within the specified epsilon
        perturbation = np.random.uniform(-epsilon, epsilon, input_shape)
        
        # Compute the difference between perturbed and original outputs
        y1 = model(x, training=False)
        y2 = model(x + perturbation, training=False)
        difference = np.linalg.norm(y2 - y1)
        
        # Update the Lipschitz constant
        lipschitz_constant = max(lipschitz_constant, difference / np.linalg.norm(perturbation))
    
    return lipschitz_constant
