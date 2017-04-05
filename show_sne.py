"""Visualise the low dimensional embedding"""

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def show_sne():
    Y = np.loadtxt('mnist2500_Ytsne.txt')
    labels = np.loadtxt('mnist2500_labels.txt')
    plt.scatter(Y[:, 0], Y[:, 1], 20, labels)


def show_sphere_sne():
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d', aspect='equal')
    Y = np.loadtxt('mnist2500_Y.txt')
    labels = np.loadtxt('mnist2500_labels.txt')
    ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c=labels)


if __name__ == "__main__":
    show_sne()
    show_sphere_sne()
    plt.show()
