import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from keras import backend as K
from keras.layers import Input, Dense, Lambda, Layer, Add, Multiply
from keras.models import Model, Sequential
from keras.datasets import mnist
import tensorflow as tf

def vae_model(original_dim,epsilon_std = 1.0):
    #original_dim = 4158
    middle_dim = 1000
    end_dim = 250
    latent_dim = 2
    #batch_size = 256
    #epochs = 250
    #epsilon_std = 1.0

    def nll(y_true, y_pred):
        return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)

    class KLDivergenceLayer(Layer):
        def __init__(self, *args, **kwargs):
            self.is_placeholder = True
            super(KLDivergenceLayer, self).__init__(*args, **kwargs)
        def call(self, inputs):
            mu, log_var = inputs
            kl_batch = - .5 * K.sum(1 + log_var -K.square(mu) - K.exp(log_var), axis=-1)
            self.add_loss(K.mean(kl_batch), inputs=inputs)
            return inputs


    decoder = Sequential([
    Dense(end_dim, input_dim=latent_dim, activation='relu'),
    Dense(middle_dim, activation='relu'),
    Dense(original_dim, activation='sigmoid')
    ])

    x = Input(shape=(original_dim,))
    h = Dense(middle_dim, activation='relu')(x)
    h = Dense(end_dim, activation='relu')(h)
    z_mu = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)
    z_mu, z_log_var = KLDivergenceLayer()([z_mu, z_log_var])
    z_sigma = Lambda(lambda t: K.exp(.5*t))(z_log_var)
    eps = Input(tensor=K.random_normal(stddev=epsilon_std,shape=(K.shape(x)[0], latent_dim)))
    z_eps = Multiply()([z_sigma, eps])
    z = Add()([z_mu, z_eps])
    x_pred = decoder(z)

    vae = Model(inputs=[x, eps], outputs=x_pred)
    vae.compile(optimizer='rmsprop', loss=nll)
    return vae

epochs = [600,700,800,900,1000]
total_smiles_gen = len(final)*len(epochs)
smiles_list = []*total_smiles_gen
x = 0
for epoch in epochs:
    print("Generation iteration number: {}".format(x))
    x = x + 1
    batch_size = 256
    model = vae_model(original_dim = 4158, epsilon_std = 1)
    model.fit(train,train,shuffle=True,epochs=epoch,batch_size=batch_size, verbose = 0)
    for i in range(len(final)):
        pred = model.predict(final[i:i+1])
        smiles_list.append(to_smiles(pred, smiles[i], tokens))
