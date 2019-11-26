import matplotlib.pyplot as plt
import numpy as np
import pickle
with open('../mfcc_test_files/mfcc_dict3.pickle', 'rb') as handle:
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

mean_X = np.mean(X)
std_X = np.std(X)

np.savez_compressed('normalization_data.npz', mean_X, std_X)

# print(mean_X)
# print(std_X)

X = (X - mean_X) / std_X

plt.pcolormesh(X[20,4])
plt.show()

# from utils import train_and_validate
#
# model_type = 'conv'
#
# models, acc = train_and_validate(X, y, model_type=model_type, n_splits=8,  epochs=30, data_shuffle=False)
#
# best_model = models[np.argmax(acc)]
# model_name = 'model_'+model_type+'_trained.h5'
# best_model.save(model_name)

