from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib import cm
# from matplotlib.colors import ListedColormap
import random
# colormap = cm.get_cmap('tab20')
# cm_dark = ListedColormap(colormap.colors[::2])

plt.ion()
# functions
# function to take input of data and number of clusters, return centroids and other data
def get_random_centroids(data_points, n_centroids=2):
  centroids_idx = random.sample(list(range(len(data_points))), n_centroids)
  return data_points[centroids_idx], np.array([x for i,x in enumerate(data_points) if i not in centroids_idx])

# function to group data according to centroids
def group_to_centroids(data_points, centroids):
  group = []
  for x in data_points:
    distances = [((x - c)**2).sum()**0.5 for c in centroids]
    group.append(np.argmin(distances))
  return group

# function to calculate centroids from grouped data
def find_centroids(data_points, groups):
  return np.array([data_points[groups == g].mean(axis=0) for g in np.unique(groups)])    

# generate data
data = make_blobs(n_samples=200, n_features=2, centers=4, cluster_std=1.6, random_state=50)

points = data[0]

plt.scatter(data[0][:,0], data[0][:,1], c=data[1])

# plt.show()
# input()

# identify initial centroids
centroids, others = get_random_centroids(points, 4)
# plt.scatter(centroids[:,0], centroids[:,1], marker='*', c='k')

diff_centroids = np.array([np.inf, np.inf])
previous_centroids = np.array([np.inf, np.inf])
index = 0
fig = plt.figure()
ax = fig.add_subplot(111)
# repeat until centroids stabilise
while np.any(diff_centroids > 0.05):
  index += 1
  ## group data to centroids
  groups = group_to_centroids(others, centroids)

  ## update centroids
  centroids = find_centroids(others, groups)
  print(index, centroids)

  ## visualise current clusters and centroids
  ax.clear()
  ax.scatter(others[:,0], others[:,1], c=groups)
  ax.scatter(centroids[:,0], centroids[:,1], marker='*', c='k')  

  diff_centroids = np.abs(centroids - previous_centroids)
  previous_centroids = centroids
  plt.pause(1)


# plt.figure()
# plt.scatter(data[0][:,0], data[0][:,1], c=data[1], cmap=cm_dark)
# plt.scatter(centroids[:,0], centroids[:,1], marker='*', c='k')  

## visualise the data and centroids

# plt.show()
#input()

# repeat the clustering process with different initial centroids
