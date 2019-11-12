from sklearn.mixture import GaussianMixture
import pickle


def model(X, id, number, n_comp, n_iter):

    gmm = GaussianMixture(n_components = n_comp, covariance_type='diag', max_iter=n_iter)

    for i in range(id):
        gmm.fit(X[i][number])

    return gmm


#model_ = model(mfcc_dict, 18, 3, 6, 10)