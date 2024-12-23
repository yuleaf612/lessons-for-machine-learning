import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# generate sample data
np.random.seed(0)
X, y = datasets.make_moons(200, noise=0.20)

y_true = np.array(y).astype(float)


# generate nn output target
t = np.zeros((X.shape[0], 2))
t[np.where(y==0), 0] = 1
t[np.where(y==1), 1] = 1

#plt.scatter(X[:,0],X[:,1],c=y,cmap=plt.cm.Spectral)
#plt.show()

class NN_Model:
    epsilon = 0.01  # learning rate
    n_epoch = 1000  # iterative number


nn = NN_Model()
nn.n_input_dim = X.shape[1]  # input size
nn.n_hide_dim = 8  # hidden node size
nn.n_output_dim = 2  # output node size

# initial weight array
nn.W1 = np.random.randn(nn.n_input_dim, nn.n_hide_dim) / np.sqrt(nn.n_input_dim)
nn.b1 = np.zeros((1, nn.n_hide_dim))
nn.W2 = np.random.randn(nn.n_hide_dim, nn.n_output_dim) / np.sqrt(nn.n_hide_dim)
nn.b2 = np.zeros((1, nn.n_output_dim))


# define sigmod & its derivate function
def sigmod(x):
    return 1.0 / (1 + np.exp(-x))


# network forward calculation
def forward(n, x):
    n.z1 = sigmod(x.dot(n.W1) + n.b1)
    n.z2 = sigmod(n.z1.dot(n.W2) + n.b2)
    return n


def backpropagation(n, x, t):
    for i in range(n.n_epoch):
        # forward to calculate each node's output
        forward(n, x)

        # print loss, accuracy
        L = np.sum((n.z2 - t) ** 2)

        y_pred = np.argmax(nn.z2, axis=1)
        acc = accuracy_score(y_true, y_pred)

        if i % 100 == 0:
            print("epoch [%4d] L = %f, acc = %f" % (i, L, acc))

        # calc weights update
        d2 = n.z2 * (1 - n.z2) * (t - n.z2)
        d1 = n.z1 * (1 - n.z1) * (np.dot(d2, n.W2.T))

        # update weights
        n.W2 += n.epsilon * np.dot(n.z1.T, d2)
        n.b2 += n.epsilon * np.sum(d2, axis=0)
        n.W1 += n.epsilon * np.dot(x.T, d1)
        n.b1 += n.epsilon * np.sum(d1, axis=0)


nn.n_epoch = 2000
backpropagation(nn, X, t)

#plt.scatter(X[:,0], X[:,1], c=y_pred, cmap=plt.cm.Spectral)
#plt.show()

# plot data
y_pred = np.argmax(nn.z2, axis=1)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
plt.title("ground truth")
plt.show()

plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap=plt.cm.Spectral)
plt.title("predicted")
plt.show()