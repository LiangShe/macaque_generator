# -*- coding: utf-8 -*-

#%% visualize latent space, for latent dimension larger than 2

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from keras.datasets import mnist
from VAE import VAE_net


#%% load test data
img_size = (28, 28, 1)


_, (x_test, y_test) = mnist.load_data()

x_test = x_test.astype('float32') / 255.
x_test = x_test.reshape((x_test.shape[0],) + img_size)


#%% load trained VAE model

model_name = './trained_models/vae_model_latent_5'
vae_net = VAE_net.load(model_name)


#%% 2D scatter plot of test data for each pair of latent dimensions

latent_dim = vae_net.config.latent_dim
batch_size = vae_net.config.batch_size

x_test_encoded = vae_net.encoder.predict(x_test, batch_size=batch_size)

plt.figure(figsize=(8, 8))
for i in range(latent_dim):
    for j in range(i+1,latent_dim):
        plt.subplot(latent_dim-1, latent_dim-1, i*(latent_dim-1)+j)
        plt.scatter(x_test_encoded[:, i], x_test_encoded[:, j], s=1, c=y_test)
        plt.axis('off')
        plt.suptitle('distribution of test data in latent space')
plt.show()


#%% visualize image in latent space
# one figure for each latent dimension, 
# x axis: vary value along current latent dimension, while keep other dimensions fixed
# y axis: fix the value of current latent dimension fixed, while randomly sample other dimensions 


n_x = 10 # number of image along specified latent dimension
n_y = 5 # number of image for other latent dimensions (randomly sampled)
    
def gen_images_vary_one_latent_dim(idim, n_x, n_y):
    '''generate images vary one latent dimension, 
    while keep others constant (randomly draw from Gaussian)
    '''
    z = np.zeros((batch_size, latent_dim))
    
    # linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
    # to produce values of the latent variables z, since the prior of the latent space is Gaussian
    x = norm.ppf(np.linspace(0.05, 0.95, n_x))
    y = norm.rvs(size=(n_y,latent_dim))
    
    count = 0
    for i in range(n_x):
        for j in range(n_y):
            z[count] = y[j]
            z[count,idim] = x[i]
            count += 1
            
    images = vae_net.decoder.predict(z, batch_size=batch_size)
    images = images[0:n_x*n_y]
    
    return images
    

def image_stack_to_montage(images, x, y):
    ''' '''
    h = images.shape[1]
    w = images.shape[2]
    im = np.zeros((h*y,w*x))
    c = 0
    for i in range(x):
        for j in range(y):
            im[j * h: (j + 1) * h, i * w: (i + 1) * w] = images[c].squeeze()
            c += 1
    return im
    
for idim in range(latent_dim):
    
    images = gen_images_vary_one_latent_dim(idim, n_x, n_y)
    image = image_stack_to_montage(images, n_x, n_y)

    plt.figure()
    plt.imshow(image, cmap='Greys_r')
    plt.axis('equal')
    plt.axis('off')
    plt.show()
    plt.title('latent dimension {}'.format(idim+1))
