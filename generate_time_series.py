import numpy as np


def shape_data_new(X, y_true,l_s=10,n_predictions=1):
    '''
    Shape raw input streams for ingestion into LSTM.
    l_s specifies the sequence length of prior timesteps fed into the model at each timestep t.
    Args:
        X (np array): array of input streams with dimensions [timesteps, 1, input dimensions]
        y_true (np array): spam label array

    Returns:
        X (np array): array of inputs with dimensions [timesteps, l_s, input dimensions)
        y (np array): array of outputs corresponding to true values following each sequence.
            shape = [timesteps, n_predictions, 1)
        y_label : array of outputs corresponding to labels true
    '''
    data = []
    labels = []
    for i in range(len(X) - l_s - n_predictions):
        data.append(X[i:i + l_s + n_predictions])
        labels.append(y_true[i:i + l_s + n_predictions])
    data = np.array(data)
    labels = np.array(labels)

    if len(data)==0:
        return np.array([]),np.array([]),np.array([])
    assert len(data.shape) == 3


    Xs = data[:, :-n_predictions, :]
    ys = data[:, -n_predictions,:]
    y_labels = labels[:, -n_predictions:]

    ys = ys.reshape(ys.shape[0], n_predictions, ys.shape[1])

    assert len(Xs.shape) == 3
    assert len(ys.shape) == 3
    assert len(y_labels.shape) == 2
    return Xs, ys, y_labels
 
def readDataConvertToTimeSeries(n_rest,base_directory,l_s=5,l_p=1):
    for i in range(n_rest):
        X_train_o = np.load(base_directory+"X_train_" + str(i) + ".npy")
        X_test_o = np.load(base_directory+"X_test_" + str(i) + ".npy")
        y_train_o = np.load(base_directory+"y_train_label_" + str(i) + ".npy")
        y_test_o = np.load(base_directory+"y_test_label_" + str(i) + ".npy")
        # shape, split data
        if i == 0:
            X_train, y_train, y_train_label = shape_data_new(X_train_o, y_train_o, l_s=l_s, n_predictions=l_p)
            X_test, y_test, y_test_label = shape_data_new(X_test_o, y_test_o, l_s=l_s, n_predictions=l_p)
        else:
            X_train_temp, y_train_temp, y_train_label_temp = shape_data_new(X_train_o, y_train_o, l_s=l_s,n_predictions=l_p)
            X_test_temp, y_test_temp, y_test_label_temp = shape_data_new(X_test_o, y_test_o, l_s=l_s, n_predictions=l_p)

            if len(X_train_temp)>0:
              X_train = np.concatenate((X_train, X_train_temp))
            if len(y_train_temp)>0:
              y_train = np.concatenate((y_train, y_train_temp))
            if len(X_test_temp)>0:
              X_test = np.concatenate((X_test, X_test_temp))
            if len(y_test_temp)>0:
              y_test = np.concatenate((y_test, y_test_temp))
            if len(y_train_label_temp)>0:
              y_train_label = np.concatenate((y_train_label, y_train_label_temp))
            if len(y_test_label_temp)>0:
              y_test_label = np.concatenate((y_test_label, y_test_label_temp))
        if i %10==0:
          print(i, "shapes after preperation", X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    np.save("X_test.npy", X_test)
    np.save("X_train.npy", X_train)
    np.save("y_train.npy", y_train)
    np.save("y_test.npy", y_test)
    np.save("y_train_label.npy", y_train_label)
    np.save("y_test_label.npy", y_test_label)
    print("Done")
    return
        
num_rest=193 #Yelpzip
base_directory = "./data/Time_Series_Data/Yelp_Zip/"
readDataConvertToTimeSeries(num_rest,base_directory,l_s=5,l_p=1)