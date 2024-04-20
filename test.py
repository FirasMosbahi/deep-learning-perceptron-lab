from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

X, y = datasets.make_blobs(n_samples=150, n_features=2, centers=2, cluster_std=1.05, random_state=2)

# Plotting
fig = plt.figure(figsize=(10, 8))
plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], 'r^')
plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], 'bs')
plt.xlabel("feature 1")
plt.ylabel("feature 2")
plt.title('Random Classification Data with 2 classes')


def acti_func(z):
    return int(z > 0)


def perceptron(X, y, lr, epochs):
    # X --> Inputs.
    # y --> labels/target.
    # lr --> learning rate.
    # epochs --> Number of iterations.
    # m-> number of training examples
    # n-> number of features
    m, n = X.shape
    # Initializing parameters(theta) to zeros.
    # +1 in n+1 for the bias term.
    w = np.zeros((n + 1, 1))
    # Empty list to store how many examples were
    # misclassified at every iteration.
    n_miss_list = []
    # Training.
    for epoch in range(epochs):
        # variable to store #misclassified.
        n_miss = 0
        # looping for every example.
        for idx, x_i in enumerate(X):
            # Insering 1 for bias, X0 = 1.
            x_i = np.insert(x_i, 0, 1).reshape(-1, 1)
            # Calculating prediction/hypothesis.
            y_hat = acti_func(np.dot(x_i.T, w))
            # Updating if the example is misclassified.
            if (np.squeeze(y_hat) - y[idx]) != 0:
                # Incrementing by 1.
                w = w + lr * (y[idx] - np.squeeze(y_hat)) * x_i
                n_miss += 1
        # Appending number of misclassified examples
        # at every iteration.
        n_miss_list.append(n_miss)

    return w, n_miss_list


w, n_miss_list = perceptron(X, y, 0.01, 5)


def plot_decision_boundary(X, w):
    # Extracting slope and intercept from weights
    w0, w1, w2 = w
    m = -w1 / w2
    c = -w0 / w2

    # Calculating x2 values for the decision boundary line
    x1 = np.array([np.min(X[:, 0]), np.max(X[:, 0])])
    x2 = m * x1 + c

    plt.plot(x1, x2, color='k', label='Decision Boundary')
    plt.legend()
    plt.show()


plot_decision_boundary(X, w)