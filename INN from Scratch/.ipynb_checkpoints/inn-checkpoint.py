import tensorflow as tf
import math as m
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LeakyReLU

# (logging setup below is to suppress warnings about non-existent gradients)

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# # Forward Sampling

# Priors

ANGLE = m.pi/2
LATENT_DIM = 4   # 4 angles
COND_DIM = 2     # 2 coordinates of end point


# +
def prior0(batchsize):
    return tf.expand_dims(tf.random.uniform(shape=[batchsize], minval=tf.constant(-ANGLE), maxval=tf.constant(ANGLE)),axis=-1)

def prior1(batchsize):
    return tf.expand_dims(tf.random.uniform(shape=[batchsize], minval=tf.constant(-ANGLE), maxval=tf.constant(ANGLE)),axis=-1)

def prior2(batchsize):
    return tf.expand_dims(tf.random.uniform(shape=[batchsize], minval=tf.constant(-ANGLE), maxval=tf.constant(ANGLE)),axis=-1)

def prior3(batchsize):
    return tf.expand_dims(tf.random.uniform(shape=[batchsize], minval=tf.constant(-ANGLE), maxval=tf.constant(ANGLE)),axis=-1)

def priors(bs):
    return tf.concat([prior0(bs), prior1(bs), prior2(bs), prior3(bs)], axis=-1)


# -

# Sampling functionality

def sample_forward(priors: callable, batchsize=1000, lengths: list = [1., 1., 1., 1.], show_joints=True):
    angles = priors(batchsize)
    x = tf.cumsum(tf.cos(angles) * lengths, axis=-1)
    y = tf.cumsum(tf.sin(angles) * lengths, axis=-1) 
    # add starting point
    x = tf.concat([tf.zeros(batchsize)[:, None], x], axis=-1)
    y = tf.concat([tf.zeros(batchsize)[:, None], y], axis=-1)
    if not show_joints:
        return angles, tf.concat([tf.expand_dims(x[:,-1], axis=-1),tf.expand_dims(y[:,-1], axis=-1)], axis=-1).numpy()
    return angles, tf.concat([tf.expand_dims(x, axis=-1),tf.expand_dims(y, axis=-1)], axis=-1).numpy()


angles, samples = sample_forward(priors=priors, show_joints=True)
for i in range(10):
    plt.scatter(samples[i,:,0], samples[i,:,1])
    # connect the dots
    plt.plot(samples[i,:,0], samples[i,:,1])
plt.show()


# # Helper Networks

# +
class CouplingNet(tf.keras.Model):
    def __init__(self, input_dim, dim_out):
        super().__init__()

        self.input_dim = input_dim
        self.dim_out = dim_out
        
        self.net = tf.keras.Sequential([
                tf.keras.Input(shape=self.input_dim),
                Dense(units=128, activation=LeakyReLU(alpha=0.01)),            
                Dense(units=128, activation=LeakyReLU(alpha=0.01)),
                Dense(units=2*self.dim_out, activation="linear"), #outputs s and t for each dim
            ])
        
    def call(self, inputs):
        output = self.net(inputs)
        s, t = tf.split(output, [self.dim_out, self.dim_out], axis=-1)
        return s, t
    
class Permutation(tf.keras.Model):
    def __init__(self, latent_dim):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        permutation_vec = np.random.permutation(latent_dim)
        inv_permutation_vec = np.argsort(permutation_vec)
        self.permutation = tf.Variable(initial_value=permutation_vec,
                                       trainable=False,
                                       dtype=tf.int32,
                                       name='permutation')
        self.inv_permutation = tf.Variable(initial_value=inv_permutation_vec,
                                           trainable=False,
                                           dtype=tf.int32,
                                           name='inv_permutation')
        
    def call(self, target, inverse=False):
        if not inverse:
            return tf.gather(target, self.permutation, axis=-1)
        return tf.gather(target, self.inv_permutation, axis=-1)


# -

# # INN class

class INN(tf.keras.Model):
    def __init__(self, latent_dim, cond_dim, num_layers = 6):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.split_len1 = self.latent_dim // 2
        self.split_len2 = self.latent_dim - self.split_len1
        
        self.num_layers = num_layers
        self.coupling_nets1 = [CouplingNet(self.split_len2+self.cond_dim, self.split_len1) for _ in range(self.num_layers)]
        self.coupling_nets2 = [CouplingNet(self.split_len1+self.cond_dim, self.split_len2) for _ in range(self.num_layers)]
        self.permutations = [Permutation(self.latent_dim) for _ in range(self.num_layers)]        
        
    def call(self, inputs, cond, inverse=False):
        if not inverse:
            return self._forward(inputs, cond)
        return self._inverse(inputs, cond) 
           
    def _forward(self, x, cond):
#         if inputs.shape[1] < self.latent_dim:
#              inputs = tf.concat([inputs, tf.zeros([inputs.shape[0], self.latent_dim-inputs.shape[1]])], axis=-1)
#         elif inputs.shape[1] > self.latent_dim:
#             raise ValueError("input cannot be of higher dimension than the latent dimension!")
            
        logjacdet = tf.zeros(x.shape[:-1])
        
        for layer in range(self.num_layers):
            u1, u2 = tf.split(x, [self.split_len1, self.split_len2], axis=-1)
            
            s1, t1 = self.coupling_nets1[layer](tf.concat([u2, cond], axis=-1))
            v1 = u1 * tf.exp(s1) + t1
            
            s2, t2 = self.coupling_nets2[layer](tf.concat([v1, cond], axis=-1))            
            v2 = u2 * tf.exp(s2) + t2
            
            x = tf.concat([v1, v2], axis=-1)
            
            x = self.permutations[layer](x)
            
            logjacdet = logjacdet + tf.reduce_sum(tf.concat([s1, s2], axis=-1), axis=-1)  
        z = x
        return z, logjacdet       
    
    def _inverse(self, z, cond):
        for layer in range(self.num_layers-1, -1, -1):
            z = self.permutations[layer](z, inverse=True)
            
            v1, v2 = tf.split(z, [self.split_len1, self.split_len2], axis=-1)
            
            s2, t2 = self.coupling_nets2[layer](tf.concat([v1, cond], axis=-1))
            u2 = (v2 - t2) * tf.exp(-s2)
            
            s1, t1 = self.coupling_nets1[layer](tf.concat([u2, cond], axis=-1))
            u1 = (v1 - t1) *tf.exp(-s1)
            
            z = tf.concat([u1, u2], axis=-1)
        x = z
        return x 

# # Training

# +
# mse = tf.keras.losses.MeanSquaredError()

# MMD implementation based on https://gist.github.com/MrYakobo/a77be1f0db4d7e00ed12ea95af8ccc74

# def inv_mq_kernel(x1, x2, beta = 1.0):
#     r = tf.transpose(x1)
#     r = tf.expand_dims(r, 2)
#     return tf.reduce_sum(K.exp( -beta * K.square(r - x2)), axis=-1)
  
# def MMD(x1, x2, beta):
#     """
#     maximum mean discrepancy (MMD) based on Gaussian kernel
#     function for keras models (theano or tensorflow backend)
    
#     - Gretton, Arthur, et al. "A kernel method for the two-sample-problem."
#     Advances in neural information processing systems. 2007.
#     """
#     x1x1 = inv_mq_kernel(x1, x1, beta)
#     x1x2 = inv_mq_kernel(x1, x2, beta)
#     x2x2 = inv_mq_kernel(x2, x2, beta)
#     diff = tf.reduce_mean(x1x1) - 2 * tf.reduce_mean(x1x2) + tf.reduce_mean(x2x2)
#     return diff
    

# class inn_loss(tf.keras.losses.Loss):
#     def __init__(self):
#         super().__init__()
#     def call(self, z_true, z_pred, ljd):
#         l_y = 0.5 * ((z_true[...,0]-z_pred[...,0])**2 + (z_true[...,1]-z_pred[...,1]**2))
#         l_z = z_true[...,2]**2 + z_true[...,3]**2yo
        

def forward_loss(model, x, cond):
    z_pred, ljd = model(x, cond, inverse=False)
    z_loss = 0.5 * tf.reduce_sum(z_pred**2, axis=-1)
    return tf.reduce_sum(z_loss - ljd)

def grad(model, x, cond):
    with tf.GradientTape() as tape:
        loss_value = forward_loss(model, x, cond)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


# -

# Initialize INN until it produces non-NaN outputs:

def initialize_INN(INN, latent_dim=LATENT_DIM, cond_dim=COND_DIM):    
    trainingINN = INN(latent_dim=latent_dim, cond_dim=cond_dim)

    angles, points = sample_forward(priors=priors, lengths=[1., 1., 1., 1.], batchsize=1000, show_joints=False)

    while np.isnan(forward_loss(trainingINN, angles, points).numpy()):
        trainingINN = INN(latent_dim=latent_dim, cond_dim=cond_dim)
        
    return trainingINN


trainingINN = initialize_INN(INN)


def train_INN(batch_size = 1000, epochs = 1000, init=True, pretrained_model=None):
    if init:
        inn = initialize_INN(INN)
    else: inn = pretrained_model
    
    train_loss_results = []
    train_accuracy_results = []

    training_optimizer = tf.keras.optimizers.Adam()
    
    angles, points = sample_forward(priors=priors, lengths=[1., 1., 1., 1.], batchsize=1000, show_joints=False)

    loss_value, grads = grad(inn, angles, points)
    print(f"Step {training_optimizer.iterations.numpy()}, Initial Loss: {loss_value.numpy()}")
    min_loss=0
    best_model = inn

    for epoch in range(epochs):
        if np.isnan(forward_loss(inn, angles, points).numpy()):
            return best_model
        angles, points = sample_forward(priors=priors, lengths=[1., 1., 1., 1.], batchsize=batch_size, show_joints=False)

        loss_value, grads = grad(inn, angles, points)
        if loss_value < min_loss: 
            min_loss = loss_value
            best_model = inn

        training_optimizer.apply_gradients([
                                        (gradient, var) 
                                        for (gradient, var) in zip(grads, inn.trainable_variables) 
                                        if grad is not None])

        if training_optimizer.iterations.numpy()%(epochs/10) == 0:
            print(f"Step {training_optimizer.iterations.numpy()},         Loss: {forward_loss(inn, angles, points).numpy()}")

        train_loss_results.append(forward_loss(inn, angles, points).numpy())
        plt.plot(train_loss_results)
        plt.draw()
        
    return best_model


loss, test_model = train_INN()

test_model

# # Testing

# +
angles, points = sample_forward(priors=priors, lengths=[1., 1., 1., 1.], batchsize=1, show_joints=False)

print(angles.shape)
test_model = loss[1]
prediction = test_model(inputs=tf.random.normal((1,4)), cond=points, inverse=False)[0]

print(prediction.numpy()[0, :2], points[0])
plt.scatter(prediction.numpy()[0,0], prediction.numpy()[0,1])
plt.scatter(points[0,0], points[0,1])
plt.xlim(0, 4)
plt.ylim(-4, 4)
plt.plot()


# -
# # Testing Backward Process

def plot_reconstructed_angles(angles, lengths=[1.,1.,1.,1.], batchsize=1000):
    x = tf.cumsum(tf.cos(angles) * lengths, axis=-1)
    y = tf.cumsum(tf.sin(angles) * lengths, axis=-1) 
    # add starting point
    x = tf.concat([tf.zeros(batchsize)[:, None], x], axis=-1)
    y = tf.concat([tf.zeros(batchsize)[:, None], y], axis=-1)
    return tf.concat([tf.expand_dims(x, axis=-1),tf.expand_dims(y, axis=-1)], axis=-1).numpy()


# +
angles, points = sample_forward(priors=priors, lengths=[1., 1., 1., 1.], batchsize=1)
points = points[0]

prediction = trainingINN(angles)
angles_reconstructed = trainingINN(angles, inverse=True)

points_reconstructed = plot_reconstructed_angles(angles_reconstructed, batchsize=1)[0]

plt.plot(points[:,0], points[:,1])
plt.plot(points_reconstructed[:,0], points_reconstructed[:,1])
# -

# # Save Weights

# +
# trainingINN.save_weights('./checkpoints/my_checkpoint')
# -

# # Load Weights

# +
# trainingINN.load_weights('./checkpoints/my_checkpoint')
# -


