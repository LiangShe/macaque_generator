# -*- coding: utf-8 -*-

'''This script demonstrates how to build a variational autoencoder
with Keras and deconvolution layers.

Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
'''


from keras.layers import Input, Dense, Lambda, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras import metrics

class vae_config:
    
    img_size = (28, 28)
    img_chns = 1
    batch_size = 100
    filter_num = 64
    kernel_size = 3
    latent_dim = 2
    intermediate_dim = 128
    epsilon_std = 1.0
    

def encoder_layers(x, c):
    
    
    x = Conv2D(c.img_chns, (2, 2), padding='same', activation='relu', name='conv_1')(x)
    x = Conv2D(c.filter_num, (2, 2), padding='same', activation='relu', name='conv_2', strides=(2, 2))(x)
    x = Conv2D(c.filter_num, c.kernel_size, padding='same', activation='relu', name='conv_3')(x)
    x = Conv2D(c.filter_num, c.kernel_size, padding='same', activation='relu', name='conv_4')(x)
    x = Flatten(name = 'encoder_reshape')(x)
    x = Dense(c.intermediate_dim, activation='relu', name='encoder_dense_1')(x)
    
    z_mean = Dense(c.latent_dim, name='encoder_mean')(x)
    z_log_var = Dense(c.latent_dim, name='encoder_var')(x)
    
    return z_mean, z_log_var


def decoder_layers(x, c):
    
    if K.image_data_format() == 'channels_first':
        output_shape = (c.batch_size, c.filter_num, 14, 14)
    else:
        output_shape = (c.batch_size, 14, 14, c.filter_num)
        
    x = Dense(c.intermediate_dim, activation = 'relu', name='decoder_dense_1')(x)
    x = Dense(c.filter_num * 14 * 14, activation = 'relu', name='decoder_dense_2')(x)

    x = Reshape(output_shape[1:], name='decoder_reshape')(x)
    x = Conv2DTranspose(c.filter_num, c.kernel_size, padding='same', activation='relu', name='deconv_1')(x)
    x = Conv2DTranspose(c.filter_num, c.kernel_size, padding='same', activation='relu', name='deconv_2')(x)
    x = Conv2DTranspose(c.filter_num, c.kernel_size, padding='valid', activation = 'relu', strides = (2, 2), name='deconv_3')(x)
    img_decoded = Conv2D(c.img_chns, kernel_size = 2, padding='valid', activation='sigmoid', name='deconv_4')(x)
    
    return img_decoded


def vae_model( config = vae_config() ):
    '''return vae, encoder, decoder '''
    #
    img_size = config.img_size
    img_chns = config.img_chns
    batch_size = config.batch_size
    latent_dim = config.latent_dim
    epsilon_std = config.epsilon_std
    
    if K.image_data_format() == 'channels_first':
        img_shape = (img_chns,) + img_size
    else:
        img_shape = img_size + (img_chns,)
    
    #%% encoder
    img_input = Input(batch_shape = (batch_size,) + img_shape)
    z_mean, z_log_var = encoder_layers(img_input, config)
    encoder = Model(img_input, z_mean, name = 'encoder')
    
    #%% decoder
    latent_input = Input(shape=(latent_dim,))
    img_decoded = decoder_layers(latent_input, config)
    decoder = Model(latent_input, img_decoded, name = 'decoder')
    
    
    #%% sampling
    def sampling(args):
        z_mean, z_log_var = args
        
        epsilon = K.random_normal(shape=(batch_size, latent_dim),
                                  mean=0., stddev=epsilon_std)
        return z_mean + K.exp(z_log_var) * epsilon
    
    # note that "output_shape" isn't necessary with the TensorFlow backend
    # so you could write `Lambda(sampling)([z_mean, z_log_var])`
    z = Lambda(sampling, output_shape=(latent_dim,), name = 'sampling')([z_mean, z_log_var])
    
    #%% vae model for training
    vae_output = decoder(z)
    vae = Model(img_input, vae_output)
    
    #%% loss 
    def vae_loss():
        img_flatten = K.flatten(img_input)
        img_decoded_flatten = K.flatten(vae_output)
        xent_loss = img_size[0] * img_size[1] * metrics.binary_crossentropy(img_flatten, img_decoded_flatten)
        kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)
    
    vae.add_loss(vae_loss())

    
    return vae, encoder, decoder