import numpy as np

def load_numpy_arrays():
    directory = "./data/"
    X_train = np.load(directory+"X_train.npy")
    y_train = np.load(directory+"y_train.npy")

    X_test = np.load(directory+"X_test.npy")
    y_test = np.load(directory+"y_test.npy")

    print("Data loaded", X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    return X_train, y_train, X_test, y_test

def load_numpy_arrays_withoutXtrain():
    directory = "./data/"
    y_train = np.load(directory+"y_train.npy")

    X_test = np.load(directory+"X_test.npy")
    y_test = np.load(directory+"y_test.npy")

    print("Data loaded",y_train.shape, X_test.shape, y_test.shape)
    return y_train, X_test, y_test
    
def load_labels():
    directory = "./data/"
    y_test_label = np.load(directory+"y_test_label.npy")

    y_train_label = np.load(directory+"y_train_label.npy")

    print("Data loaded", y_train_label.shape,y_test_label.shape)
    return y_train_label,y_test_label

