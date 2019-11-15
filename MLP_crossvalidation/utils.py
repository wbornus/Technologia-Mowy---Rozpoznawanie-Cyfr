def train_and_validate(dict_X, dict_y, model_type='dense', data_shuffle = False,  n_splits=5, epochs=10):
    """
    params:
    dict_X: dictionary with training data
    dict_y: dictionary with training labels
    model_type: type of model to be trained (dense, conv2d)
    n_splits: number defining division of data eg. n_split=5 - division = 80% / 20%
    epochs: number training epochs
    trains models with data division specified in n_splits
    evaluates models - prints accuracy of each trained model
    returns: list of all rained models
    """
    from sklearn.model_selection import KFold
    from keras.models import load_model
    from keras.utils import to_categorical
    import numpy as np
    import matplotlib.pyplot as plt
    x_valid = KFold(n_splits)
    model_list = []
    acc_list = []

    if data_shuffle:
        random_idxs = np.random.permutation(len(dict_X))
        dict_X = dict_X[random_idxs]
        dict_y = dict_y[random_idxs]

    for train_idxs, test_idxs in x_valid.split(dict_X):
        model = load_model('model_'+model_type+'_untrained.h5')
        X_train = dict_X[train_idxs, :]
        X_test = dict_X[test_idxs, :]
        y_train = dict_y[train_idxs, :]
        y_test = dict_y[test_idxs, :]
        y_train_onehot = to_categorical(np.reshape(y_train, (y_train.shape[0] * y_train.shape[1])), 10)
        y_test_onehot = to_categorical(np.reshape(y_test, (y_test.shape[0] * y_test.shape[1])), 10)

        if model_type == 'dense':
            X_train_flat = np.reshape(X_train, (X_train.shape[0]*X_train.shape[1], 70*13))
            X_test_flat = np.reshape(X_test, (X_test.shape[0]*X_test.shape[1], 70*13))
            model.fit(X_train_flat, y_train_onehot, epochs=epochs,
                        validation_data=(X_test_flat, y_test_onehot))
            model_list.append(model)
            acc_list.append(model.evaluate(X_test_flat, y_test_onehot)[1])

        if model_type == 'conv' or model_type == 'convolutional':
            X_train_conv = np.reshape(X_train, (X_train.shape[0]*X_train.shape[1], 70, 13, 1))
            X_test_conv = np.reshape(X_test, (X_test.shape[0]*X_test.shape[1], 70, 13, 1))
            model.fit(X_train_conv, y_train_onehot, epochs=epochs, steps_per_epoch=10,
                      validation_data=(X_test_conv, y_test_onehot), validation_steps=1)
            model_list.append(model)
            acc_list.append(model.evaluate(X_test_conv, y_test_onehot)[1])

    plt.plot(acc_list)
    title_model_type = 'model type: ' + str(model_type) +'  ||||  '
    title_data_shuffle = 'data shuffle: ' + str(data_shuffle) + '\n'

    title = 'Acc of each model\n' + title_model_type + title_data_shuffle \
            + 'mean: ' \
            + str(np.mean(acc_list)) + \
            '  ||||  ' + 'std_dev ' + str(np.std(acc_list))
    plt.ylim(0, 1.2)
    plt.grid()
    plt.title(title)
    plt.ylabel('accuracy [%]')
    plt.xlabel('model')
    plt.show()

    return model_list
