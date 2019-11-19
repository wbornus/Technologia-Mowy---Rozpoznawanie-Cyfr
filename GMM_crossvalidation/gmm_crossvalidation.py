from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import pickle

with open('mfcc_dict.pickle', 'rb') as handle:
    mfcc_dict = pickle.load(handle)


def gmm_function(MFCC, n_number, n_comp, n_iter, n_person):
    GMM = GaussianMixture(n_components=n_comp,
                                  covariance_type="spherical",
                                  tol=0.001, reg_covar=1e-06,
                                  max_iter=n_iter, n_init=1,
                                  init_params='kmeans',
                                  weights_init=None,
                                  means_init=None,
                                  precisions_init=None,
                                  random_state=None,
                                  warm_start=False,
                                  verbose=0,
                                  verbose_interval=10)
    for n in n_person:
        GMM.fit(MFCC[n][n_number])
    return GMM




xvalid = KFold(n_splits=5)

sum = 0
max_number = 10
n_iter = 20
n_comp = 1

for train, test in xvalid.split(mfcc_dict):
    gmm_array = []
    acc = 0

    print(train, test)
    for i in range(0, 10):
        gmm_array.append(gmm_function(mfcc_dict, i, n_comp, n_iter, train))


    for t in test:
        for k in range(0, max_number):
            tmp = []
            for j in range(0, max_number):
                tmp.append(gmm_array[j].score(mfcc_dict[t][k]))
            print(tmp.index(max(tmp)), k)
            if max(tmp) == gmm_array[k].score(mfcc_dict[t][k]):
                acc = acc + 1
    print("accuracy for iteration " + str(test) + " = ", acc/(len(test)*max_number))
    sum = sum + acc/(len(test)*max_number)


print("Overall accuracy: ", sum/xvalid.n_splits)
