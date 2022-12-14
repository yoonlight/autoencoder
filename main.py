from data.mnist import load_data
from model.Autoencoder import Autoencoder
from model.DeepAutoencoder import DeepAutoencoder
from util.plot_img import plot_image


latent_dim = 32


autoencoder = Autoencoder(latent_dim)
autoencoder = DeepAutoencoder(latent_dim)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

x_train, x_test = load_data()

autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test))

decoded_imgs = autoencoder.predict(x_test)

plot_image(x_test, decoded_imgs)
