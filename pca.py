"""Principal component analysis for t-SNE"""

import numpy as np
from sklearn.decomposition import PCA


def pca(X):
    pca = PCA(n_components=50)
    pca.fit(X)
    low = pca.fit_transform(X)
    return low


if __name__ == '__main__':
    X = np.loadtxt('mnist2500_X.txt.bz2')
    print('Original data has shape {}'.format(X.shape))
    low = pca(X)
    print('PCA version has shape {}'.format(low.shape))
    np.savetxt('mnist2500_pca.txt', low)
