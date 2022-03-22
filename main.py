import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# a quick implementation of linear regression but with usage of kernel method


def kernel_function(x, z):
    # simple 1-d gaussian kernel
    gamma = 100
    return np.exp(- ((x - z) ** 2 / gamma))


def generate_kernel_matrix(x):
    # generate m x m kernel matrix
    n = x.shape[0]
    k = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            k[i, j] = kernel_function(x[i, :], x[j, :])
    return k


def train(x, y, learning_rate, steps):
    beta = np.zeros((x.shape[0], 1))
    kernel_matrix = generate_kernel_matrix(x)
    labels = y.reshape(len(y), 1)

    for i in range(steps):
        beta += learning_rate * (labels - kernel_matrix.dot(beta))

    plt.scatter(x, y)

    axes = plt.gca()
    (x_min, x_max) = axes.get_xlim()

    y_a = np.empty(0)
    x_a = range(int(x_min), int(x_max))
    # Theta.transpose * feature_mapping(x) = sum beta_i * K(x_i, x)
    for i in x_a:
        s = 0
        for j in range(x.shape[0]):
            s += beta[j] * kernel_function(x[j], i)
        y_a = np.append(y_a, s)

    # plot approximated resulting curve as straight lines between segments
    for i in range(len(y_a) - 1):
        plt.plot([x_a[i], x_a[i]+1], [y_a[i], y_a[i+1]], color='r')

    plt.show()


def main():
    data = pd.read_csv('data.csv').dropna()
    train(data.iloc[:, :-1].to_numpy(), data.iloc[:, -1].to_numpy(), 0.001, 1000)


if __name__ == '__main__':
    main()
