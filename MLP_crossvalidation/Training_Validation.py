import matplotlib.pyplot as plt
import numpy as np
import pickle
with open('../mfcc_loader/mfcc_dict.pickle', 'rb') as handle:
        mfcc_dict = pickle.load(handle)

"""data preprocessing"""
X = []
y = []
to_avg = []
for it_0 in range(len(mfcc_dict)):
        X_column = []
        y_column = []
        for it_1 in range(len(mfcc_dict[0])):
                tmp = np.array(mfcc_dict[it_0][it_1])
                to_avg.append(tmp.shape[0])
                if tmp.shape[0] > 70:
                        tmp = tmp[:70, :]
                else:
                        to_append = np.zeros((70 - tmp.shape[0], 13))
                        tmp = np.concatenate((tmp, to_append), axis=0)
                X_column.append(tmp)
                y_column.append(it_1)
        X.append(X_column)
        y.append(y_column)

X = np.array(X)
y = np.array(y)


print(np.mean(to_avg))
print(np.std(to_avg))
print(y[10, 5])
print(X.shape)

from utils import train_and_validate

model_type = 'dense'

models, acc = train_and_validate(X, y, model_type=model_type, epochs=30, data_shuffle=True)

best_model = models[np.argmax(acc)]
model_name = 'model_'+model_type+'_trained.h5'
best_model.save(model_name)

