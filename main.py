import numpy as np
import json
from scipy.spatial.distance import cdist
import time
import pynanoflann
import torch
import math
from scipy.io import savemat
import os


def find_closest_points(target_point, point_set):
    distance = cdist(target_point.reshape(1, -1), point_set)
    closest_indices = np.argsort(distance.flatten())[:5]
    closest_points = point_set[closest_indices]
    return closest_points, closest_indices


def calculate_rotation_norm(source_point, source_neighbours, target_point, target_neighbours):
    # Centre points around their respective origin
    centered_1 = source_neighbours - source_point
    centered_2 = target_neighbours - target_point
    # Calculate optimal rotation
    covariance_matrix = np.dot(centered_1.T, centered_2)
    U, S, VT = np.linalg.svd(covariance_matrix)
    R = np.dot(VT, U)
    # Rotate source points using optimal rotation matrix
    rotated_source_points = np.dot(centered_1, R)
    diff = centered_2 - rotated_source_points
    norm = np.linalg.norm(diff, axis=1)
    return norm

# Load Model
model = torch.load('MLP.pt')
model.eval()


with open('sparse_features.json', 'r') as f:
    data = json.load(f)

with open('dense_features.json', 'r') as f:
    dense_data = json.load(f)

# Extract all the variables
descriptors = np.transpose(np.array(data['sparse_descriptors']))
points = np.transpose(np.array(data['sparse_features']))
dense_points = np.transpose(np.array(dense_data['dense_features']))
dense_descriptors = np.transpose(np.array(dense_data['dense_descriptors']))

s_dense = dense_points[:, 2:]
t_dense = dense_points[:, :2]
dense_points = np.concatenate((s_dense, t_dense), axis=1)

original = dense_points

num_points = 20
new_features = np.zeros([num_points, 6], dtype=np.uint8)

start_time = time.time()
for i in range(math.floor(dense_descriptors.shape[0] / num_points)):
    # Create KDTree from one of the datasets
    kd_tree = pynanoflann.KDTree(n_neighbors=1, metric='L1', radius=100)
    kd_tree.fit(points[:, :2])
    # Find the closest point in data_1 for each point in data_2
    distances, indices = kd_tree.kneighbors(dense_points[:, :2])
    # Find the closest point from the dense data to inliers
    # Find the closest point from the dense data to inliers
    min_index = np.argsort(distances.flatten())[:num_points]

    # Extract the SVD of the closest 2 points
    for j in range(min_index.shape[0]):
        # Extract a dense point and find the closest points in inlier data
        source_centre = dense_points[min_index[j], 2:]
        target_centre = dense_points[min_index[j], :2]
        source_neighbour, indexes = find_closest_points(source_centre, points[:, :2])
        target_neighbour = points[indexes, 2:]

        # Calculate norm of rotated source points and target source points using SVD
        distances_norm = calculate_rotation_norm(source_centre, source_neighbour, target_centre,
                                                 target_neighbour)

        # Calculate descriptor difference
        descr_diff = np.linalg.norm(
            dense_descriptors[min_index[j], :128] - dense_descriptors[min_index[j], 128:], axis=0)

        # use the model to predict the output for the new feature
        new_feature = torch.tensor(np.hstack((descr_diff, distances_norm)), dtype=torch.float32)
        new_features[j, :] = new_feature
        new_features = torch.tensor(new_features, dtype=torch.float32)

    with torch.no_grad():
        model.eval()
        output = model(new_features)
    # If value over .5 add to the inliers

    indies = np.where(output[:, 1] > 0.70)
    points = np.vstack((points, dense_points[np.squeeze(min_index[indies]), :]))
    # Delete them from the dense points
    dense_points = np.delete(dense_points, min_index, axis=0)
    dense_descriptors = np.delete(dense_descriptors, min_index, axis=0)
    if i % 500 == 0:
        print("This is loop number: ", i)
        print("Dense Points shape: ", dense_points.shape)
        print("Points shape: ", points.shape)

end_time = time.time()
print(original.shape)
print(dense_points.shape)

mdic = {"a": points, "label": "experiment"}
savemat("matrix_00.mat", mdic)

print("Time taken: ", end_time - start_time, "seconds")
