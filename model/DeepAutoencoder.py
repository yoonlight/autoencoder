import keras
from keras import layers


class DeepAutoencoder(keras.Model):
    def __init__(self, latent_dim):
        super(DeepAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = keras.Sequential([
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(latent_dim, activation='relu'),
        ])
        self.decoder = keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(784, activation='sigmoid'),
            layers.Reshape((28, 28))
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
