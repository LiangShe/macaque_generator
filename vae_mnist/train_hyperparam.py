'''train variational autoencoder on MNIST
   for various hyperparameters
'''

from keras.datasets import mnist
from keras.callbacks import TensorBoard

from VAE import VAE_net, VAE_config

#%%
epochs = 10
batch_size = 100
output_folder = './trained_models/'

#%% load data

# input image dimensions
img_size = (28, 28, 1)

# train the VAE on MNIST digits
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_train = x_train.reshape((x_train.shape[0],) + img_size)
x_test = x_test.astype('float32') / 255.
x_test = x_test.reshape((x_test.shape[0],) + img_size)

print('x_train.shape:', x_train.shape)


#%% train model for various number of latent dimensions

for latent_dim in range(2,8):
    
    # get VAE net
    config = VAE_config()
    config.batch_size = batch_size
    config.latent_dim = latent_dim
    config.training = True
    
    vae_net = VAE_net(config)

    model_name = output_folder + 'vae_model_latent_{}'.format(latent_dim)
    
    # train
    vae_net.vae.fit(x_train,
            shuffle=True,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, x_test),
            callbacks=[TensorBoard(log_dir=model_name)]
            )
    
    # save model
    vae_net.save(model_name)
