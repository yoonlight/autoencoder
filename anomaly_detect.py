import matplotlib.pyplot as plt
import numpy as np

from data.ecg5000 import load_data
from model.AnomalyDetector import AnomalyDetector


autoencoder = AnomalyDetector()

autoencoder.compile(optimizer='adam', loss='mae')

normal_train_data, normal_test_data, test_data, anomalous_test_data = load_data()

history = autoencoder.fit(normal_train_data, normal_train_data,
                          epochs=20,
                          batch_size=512,
                          validation_data=(test_data, test_data),
                          shuffle=True)

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.show()

decoded_data = autoencoder.predict(normal_test_data)

plt.plot(normal_test_data[0], 'b')
plt.plot(decoded_data[0], 'r')
plt.fill_between(
    np.arange(140), decoded_data[0], normal_test_data[0], color='lightcoral')
plt.legend(labels=["Input", "Reconstruction", "Error"])
plt.show()

decoded_data = autoencoder.predict(anomalous_test_data)

plt.plot(anomalous_test_data[0], 'b')
plt.plot(decoded_data[0], 'r')
plt.fill_between(
    np.arange(140), decoded_data[0], anomalous_test_data[0], color='lightcoral')
plt.legend(labels=["Input", "Reconstruction", "Error"])
plt.show()
