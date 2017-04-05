"""
BEESNEES
Bregman X Ergodic Stochastic Neighbourhood Embedding X X
"""

import numpy as np


def normalise_ergodic(S):
    """Return a doubly stochastic matrix"""
    for ix in range(100):
        S = S / np.sum(S, axis=0)
        S = S / np.sum(S, axis=1)
    return S


def normalise_length(Y):
    """Normalise each row to unit length"""
    norm = np.sqrt(np.sum(Y*Y, axis=1))
    norm.shape = (len(norm), 1)
    return Y / norm


def sqr_dist(X):
    """Squared Euclidean distance, assuming X is a matrix"""
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    return D


def hbeta(D, beta=1.0):
    """Compute the perplexity and the P-row for a specific value
    of the precision of a Gaussian distribution."""
    P = np.exp(-D.copy() * beta)
    sumP = np.sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P


def adjust_precision(hdiff, beta, betamin, betamax):
    """Increase or decrease precision"""
    if hdiff > 0:
        betamin = beta
        if np.isinf(betamax):
            beta *= 2
        else:
            beta = (beta + betamax) / 2
    else:
        betamax = beta
        if np.isinf(betamin):
            beta /= 2
        else:
            beta = (beta + betamin) / 2
    return beta, betamin, betamax


def high_dim_similarity(X, perplexity):
    """Performs a binary search to get P-values in such a way that each conditional Gaussian has
 the same perplexity."""
    tol = 1e-5
    print("Computing pairwise distances...")
    (n, d) = X.shape
    D = sqr_dist(X)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    for ix in range(n):
        if ix % 500 == 0:
            print("Computing P-values for point ", ix, " of ", n, "...")

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[ix, np.concatenate((np.r_[0:ix], np.r_[ix+1:n]))]
        cur_beta = beta[ix].copy()
        (H, thisP) = hbeta(Di, cur_beta)

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:
            cur_beta, betamin, betamax = adjust_precision(Hdiff, cur_beta, betamin, betamax)
            # Recompute the values
            (H, thisP) = hbeta(Di, cur_beta)
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        beta[ix] = cur_beta
        P[ix, np.concatenate((np.r_[0:ix], np.r_[ix+1:n]))] = thisP

    # Return final P-matrix
    print("Mean value of sigma: ", np.mean(np.sqrt(1 / beta)))

    return P


def normalised_gaussian(X, perplexity=30.0, ergodic=True):
    """Return the perplexity"""
    P = high_dim_similarity(X, perplexity)
    if ergodic:
        print('Make doubly stochastic')
        P = normalise_ergodic(P)     # make P doubly stochastic
    P = 0.5*(P + P.T)
    P = P / np.sum(P)
    P = np.maximum(P, 1e-12)
    return P


def inv_sq_dist(X):
    """1/1+||x-y||^2"""
    N = X.shape[0]
    D = sqr_dist(X)
    num = (1. / (1. + D))
    num[range(N), range(N)] = 0
    return num


def low_dim_similarity(numerator):
    """Return the Student t-distribution kernel"""
    K = numerator/np.sum(numerator)
    K = np.maximum(K, 1e-12)
    return K


def kl_div(P, Q):
    """Objective function"""
    return np.sum(P * np.log(P / Q))


def grad_kl(P, Q, Y, numerator):
    """Gradient of the objective w.r.t. Y"""
    L = (P - Q) * numerator
    grad = 4 * (np.diag(np.sum(L, axis=1)) - L) @ Y
    return grad


def tsne(P, num_dim=3, max_iter=1000, initial_momentum=0.5, final_momentum=0.8, eta=500,
         min_gain=0.01):
    """Runs t-SNE on the N by N perplexity (normalised similarity) matrix P
    where N = number of examples
    """
    momentum = initial_momentum

    N = P.shape[0]
    Y = np.random.randn(N, num_dim)
    updateY = np.zeros((N, num_dim))
    gains = np.ones((N, num_dim))
    P = P * 4                          # early exaggeration

    for ix in range(max_iter):
        numerator = inv_sq_dist(Y)
        Q = low_dim_similarity(numerator)

        # Update the solution
        gradY = grad_kl(P, Q, Y, numerator)
        gains = ((gains + 0.2) * ((gradY > 0) != (updateY > 0))
                 + (gains * 0.8) * ((gradY > 0) == (updateY > 0)))
        gains[gains < min_gain] = min_gain
        updateY = momentum * updateY - eta * (gains * gradY)
        Y += updateY

        # Re-center
        Y -= np.tile(np.mean(Y, axis=0), (N, 1))

        # Increase momentum
        if ix == 20:
            momentum = final_momentum

        # Stop lying about P-values
        if ix == 100:
            P = P / 4

        if ix % 50 == 0:
            print('Iteration {}: KL(P||Q) = {}'.format(ix, kl_div(P, Q)))
    return Y


def original_tsne():
    print('Project to 2 dimensional plane')
    X = np.loadtxt('mnist2500_pca.txt')
    P = normalised_gaussian(X, ergodic=False)
    Y = tsne(P, num_dim=2)
    np.savetxt('mnist2500_Ytsne.txt', Y)


def beesnees():
    print('Project to sphere')
    X = np.loadtxt('mnist2500_pca.txt')
    P = normalised_gaussian(X)
    Y = tsne(P)
    Y = normalise_length(Y)
    np.savetxt('mnist2500_Y.txt', Y)


if __name__ == '__main__':
    original_tsne()
    beesnees()
