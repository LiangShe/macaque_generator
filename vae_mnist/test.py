# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from keras.datasets import mnist
from VAE import vae_model


#%%
img_size = (28, 28, 1)
batch_size = 100

# train the VAE on MNIST digits
_, (x_test, y_test) = mnist.load_data()


x_test = x_test.astype('float32') / 255.
x_test = x_test.reshape((x_test.shape[0],) + img_size)


#%% 

vae, encoder, generator = vae_model()

vae.load_weights('vae_weights.h5')

#encoder.load_weights('vae_weights.h5', by_name = True)
#generator.load_weights('vae_weights.h5', by_name = True)


# display a 2D plot of the digit classes in the latent space
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(6, 3))
plt.subplot(1,2,1)
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
plt.colorbar()



# display a 2D manifold of the digits
n = 15  # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
# linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
# to produce values of the latent variables z, since the prior of the latent space is Gaussian
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)
        x_decoded = generator.predict(z_sample, batch_size=batch_size)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.subplot(1,2,2)
plt.imshow(figure, cmap='Greys_r')
plt.show()