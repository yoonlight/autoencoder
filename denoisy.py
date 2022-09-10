from data.noisy_mnist import load_data
from model.ConvAutoencoder import ConvAutoencoder
from util.plot_img import plot_image


autoencoder = ConvAutoencoder()
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')


x_train, x_test = load_data()

autoencoder.fit(x_train, x_train,
                epochs=10,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test))

decoded_imgs = autoencoder.predict(x_test)

plot_image(x_test, decoded_imgs)
