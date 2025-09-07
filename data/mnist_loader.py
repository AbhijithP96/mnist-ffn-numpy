import tensorflow as tf

def get_data() -> tuple:
    """Load MNIST dataset using TensorFlow.
    Returns:
        Tuple of training and testing datasets.
    """

    (x_train, y_train), (x_, y_) = tf.keras.datasets.mnist.load_data()

    x_dev = x_[:x_.shape[0]//2]
    y_dev = y_[:y_.shape[0]//2]

    x_test = x_[x_.shape[0]//2:]
    y_test = y_[y_.shape[0]//2:]

    return (x_train, y_train), (x_test, y_test), (x_dev, y_dev)

def preprocess_data(train, dev, test):

    (x_train, y_train) = train
    (x_dev, y_dev) = dev
    (x_test, y_test) = test

    # Normalize the images to [0, 1] range
    x_train = x_train.astype("float32") / 255.0
    x_dev = x_dev.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Reshape x to (num_samples, dimension*dimension)
    x_train = x_train.reshape((x_train.shape[0], -1))
    x_dev = x_dev.reshape((x_dev.shape[0], -1))
    x_test = x_test.reshape((x_test.shape[0], -1))

    # one hot encode the labels
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    y_dev = tf.keras.utils.to_categorical(y_dev, num_classes=10)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

    return (x_train, y_train), (x_dev, y_dev) ,(x_test, y_test)