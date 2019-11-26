
from sklearn.model_selection import KFold
from keras.models import load_model
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from python_speech_features import mfcc


def evaluate(X_eval, y_eval, model_type='conv'):

    model = load_model('../MLP_crossvalidation/model_'+model_type+'_trained.h5')

    y_eval_onehot = to_categorical(np.reshape(y_eval, (y_eval.shape[0] * y_eval.shape[1])), 10)

    if model_type == 'dense':
        X_eval_flat = np.reshape(X_eval, (X_eval.shape[0]*X_eval.shape[1], 70*13))
        score, acc = model.evaluate(X_eval_flat, y_eval_onehot)
    elif model_type == 'conv':
        X_eval_conv = np.reshape(X_eval, (X_eval.shape[0]*X_eval.shape[1], 70, 13, 1))
        score, acc = model.evaluate(X_eval_conv, y_eval_onehot)
    else:
        return -1
    
    return score, acc


def predict_digit(data, fs=16000, model_type='conv'):
    """
    :param data: raw wave file
    :param fs: sampling frequency of data
    :param model_type: type of model
    :return: predicttion of digit said in raw data file
    """
    model_directory = '../MLP_crossvalidation/'
    data = np.array(data)
    data_mfcc = mfcc(data, samplerate=fs, winlen=0.025,
                    winstep=0.01, numcep=13, nfilt=26, nfft=512, lowfreq=0, highfreq=None,
                    preemph=0.97, ceplifter=22, appendEnergy=True, winfunc=np.hamming)

    if data_mfcc.shape[0] > 70:
        data_mfcc = data_mfcc[:70, :]
    else:
        to_append = np.zeros((70 - data_mfcc.shape[0], 13))
        data_mfcc = np.concatenate((data_mfcc, to_append), axis=0)

    if model_type == 'conv':
        input_data = np.reshape(data_mfcc, (1, 70, 13, 1))
        model_name = 'model_'+model_type+'_trained.h5'
        model = load_model(model_directory+model_name)
        prediction = np.reshape(np.argmax(model.predict(input_data), axis=1), (1,))

    elif model_type == 'dense':
        input_data = np.reshape(data_mfcc, (1, 70*13))
        model_name = 'model_'+model_type+'_trained.h5'
        model = load_model(model_directory+model_name)
        prediction = np.reshape(np.argmax(model.predict(input_data), axis=1), (1,))

    else:
        return -1

    return prediction[0]
