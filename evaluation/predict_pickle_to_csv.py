import pickle
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
import os
import csv
import pandas as pd

with open('./mfcc_eval/mfcc_dict_eval.pickle', 'rb') as handle:
    mfcc_dict = pickle.load(handle)


model = load_model('../MLP_crossvalidation/model_conv_trained.h5')

fnames = os.listdir('../eval/')

mfcc_data = []
for it in range(len(mfcc_dict)):
    tmp = np.array(mfcc_dict[it])
    if tmp.shape[0] > 90:
        tmp = tmp[:90, :]
    else:
        to_append = np.zeros((90 - tmp.shape[0], 13))
        tmp = np.concatenate((tmp, to_append), axis=0)
    mfcc_data.append(tmp.tolist())

normalization_data = np.load('../MLP_crossvalidation/normalization_data.npz')
mean_X, std_X = [normalization_data[f] for f in normalization_data.files]
print(mean_X, std_X)

mfcc_data = np.array(mfcc_data)
print(mfcc_data.shape)
mfcc_data = (mfcc_data - mean_X) / std_X

mfcc_data_input = np.expand_dims(mfcc_data, axis=3)
print(mfcc_data_input.shape)

onehot_predictions = model.predict(mfcc_data_input)
predictions = np.argmax(onehot_predictions, axis=1)
scores = np.array([])
for it in range(len(predictions)):
    score = np.log(onehot_predictions[it, predictions[it]])
    scores = np.append(scores, score)

# print(predictions)
# print(scores)

for fname, prediction, score in zip(fnames, predictions, scores):
    print(fname, ',', end='')
    print(prediction, ',', end='')
    print(score)


data_frame = pd.DataFrame({'fnames': fnames, 'predictions': predictions, 'scores': scores})
data_frame.to_csv('results.csv', header=False, index=False)
