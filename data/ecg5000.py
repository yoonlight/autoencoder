import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

def load_data():
    # Download the dataset
    dataframe = pd.read_csv(
        'http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv', header=None)
    raw_data = dataframe.values
    # dataframe.head()

    # The last element contains the labels
    labels = raw_data[:, -1]

    # The other data points are the electrocadriogram data
    data = raw_data[:, 0:-1]

    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, test_size=0.2, random_state=21
    )

    min_val = tf.reduce_min(train_data)
    max_val = tf.reduce_max(train_data)

    train_data = (train_data - min_val) / (max_val - min_val)
    test_data = (test_data - min_val) / (max_val - min_val)

    train_data = tf.cast(train_data, tf.float32)
    test_data = tf.cast(test_data, tf.float32)

    train_labels = train_labels.astype(bool)
    test_labels = test_labels.astype(bool)

    normal_train_data = train_data[train_labels]
    normal_test_data = test_data[test_labels]

    anomalous_train_data = train_data[~train_labels]
    anomalous_test_data = test_data[~test_labels]

    return normal_train_data, normal_test_data, test_data, anomalous_test_data
