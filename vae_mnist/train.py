'''This script demonstrates how to build a variational autoencoder
with Keras and deconvolution layers.

Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
'''

from keras.datasets import mnist
from VAE import vae_model, vae_config

#%%
epochs = 5
batch_size = 100

#%% load data

# input image dimensions
img_size = (28, 28, 1)

# train the VAE on MNIST digits
(x_train, _), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_train = x_train.reshape((x_train.shape[0],) + img_size)
x_test = x_test.astype('float32') / 255.
x_test = x_test.reshape((x_test.shape[0],) + img_size)


#%%

config = vae_config()
config.batch_size = batch_size
config.latent_dim = 2

vae, _, _  = vae_model(config)


#%%
vae.compile(optimizer='rmsprop', loss=None)


print('x_train.shape:', x_train.shape)

vae.fit(x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, x_test))

vae.save_weights('vae_weights.h5')
