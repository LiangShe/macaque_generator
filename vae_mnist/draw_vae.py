# -*- coding: utf-8 -*-
from VAE import vae_model
from keras.utils import plot_model

vae, encoder, decoder  = vae_model()

plot_model(vae, to_file='./pic/vae.png', show_shapes = True)
plot_model(encoder, to_file='./pic/encoder.png', show_shapes = True)
plot_model(decoder, to_file='./pic/decoder.png', show_shapes = True)