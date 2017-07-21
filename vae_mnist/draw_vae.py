# -*- coding: utf-8 -*-
from VAE import VAE_net
from keras.utils import plot_model

vae_net = VAE_net()

plot_model(vae_net.vae, to_file='./pic/vae.png', show_shapes = True)
plot_model(vae_net.encoder, to_file='./pic/encoder.png', show_shapes = True)
plot_model(vae_net.decoder, to_file='./pic/decoder.png', show_shapes = True)