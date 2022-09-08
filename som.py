"""
Classification of four classes of gaussian points using SOM.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn_som.som import SOM


x, y = np.mgrid[-5.0:5.0:100j, -5.0:5.0:100j]
xy = np.column_stack([x.flat, y.flat])

mu1 = np.array([-2, 2])
mu2 = np.array([2, 2])
mu3 = np.array([2, -2])
mu4 = np.array([-2, -2])

sigma1 = np.array([0.5, 0.5])
sigma2 = np.array([0.8, 0.8])
sigma3 = np.array([0.5, 0.5])
sigma4 = np.array([0.8, 0.8])

covariance1 = np.diag(sigma1 ** 2)
covariance2 = 2 * np.diag(sigma2 ** 2)
covariance3 = 2 * np.diag(sigma3 ** 2)
covariance4 = np.diag(sigma4 ** 2)

z1 = 4 * multivariate_normal.pdf(xy, mean=mu1, cov=covariance1)
z2 = 6 *  multivariate_normal.pdf(xy, mean=mu2, cov=covariance2)
z3 = 4 *  multivariate_normal.pdf(xy, mean=mu3, cov=covariance3)
z4 = -5 * multivariate_normal.pdf(xy, mean=mu4, cov=covariance4)

z = z1 + z2 + z3 + z4
z = z.reshape(x.shape)

train_points1 = np.random.multivariate_normal(mu1, covariance1, 50)
train_points2 = np.random.multivariate_normal(mu2, covariance2, 50)
train_points3 = np.random.multivariate_normal(mu3, covariance3, 50)
train_points4 = np.random.multivariate_normal(mu4, covariance4, 50)

train_data = np.concatenate((train_points1, train_points2, train_points3, train_points4), axis=0)

x1 = train_data[:, 0]
y1 = train_data[:, 1]
centers_dict = {}

for index, m in enumerate([4, 12, 28, 40]):
    som = SOM(m=m, n=1, dim=2)
    prediction = som.fit_predict(train_data)
    centers = som.cluster_centers_
    centers_dict[m] = np.squeeze(centers, axis=1)
    plt.subplot(2, 2, index + 1)
    plt.scatter(x1, y1, c=prediction)
    plt.scatter(centers[:,:, 0], centers[:,:, 1], marker="*", color="black")

plt.show()