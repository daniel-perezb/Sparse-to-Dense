from scipy.spatial.distance import cdist
import numpy as np
import random
import pynanoflann
import math


def find_closest_points(target_point, point_set):
    closest_point_dist = cdist(target_point.reshape(1, -1), point_set)
    closest_indices = np.argsort(closest_point_dist.flatten())[:6]
    closest_points = point_set[closest_indices]
    mask = np.any(closest_points != target_point, axis=1)
    closest_points = closest_points[mask][:5]
    closest_indices = closest_indices[mask][:5]
    return closest_points, closest_indices


def calculate_rotation_norm(source_point, source_neighbours, target_point, target_neighbours):
    # Centre points around their respective origin
    centered_1 = source_neighbours - source_point
    centered_2 = target_neighbours - target_point
    # Calculate optimal rotation
    covariance_matrix = np.dot(centered_1.T, centered_2)
    U, S, VT = np.linalg.svd(covariance_matrix)
    R = np.dot(VT, U)
    # Rotate source source using optimal rotation matrix
    rotated_source_points = np.dot(centered_1, R)
    diff = centered_2 - rotated_source_points
    norm = np.linalg.norm(diff, axis=1)
    return norm


def descriptors_norm(descriptors, matches, source_markers, target_markers, data_augmentation=False):

    n_neighbours = 5
    # For each of the SIFT points find the closest 5 colour markers
    # Create KDTree from one of the datasets
    kd_tree = pynanoflann.KDTree(n_neighbors=n_neighbours + 1, metric='L1', radius=100)
    kd_tree.fit(source_markers)
    # Find the closest point in data_1 for each point in data_2
    distances, indices = kd_tree.kneighbors(matches[:, :2])

    # Create inliers
    distances_norm = np.zeros([distances.shape[0], 5])

    # Create outliers
    outlier_norm = np.zeros([distances.shape[0], 5])

    acceptable_distance = 40
    distance_to_original_point = 0

    for i in range(distances.shape[0]):

        if data_augmentation:
            # Create source neighbour and target neighbour array
            source_neighbour = np.zeros((5, 2))
            outlier_neighbour = np.zeros((5, 2))
            # For each of the neighbours change neighbour location if value is above a threshold
            for j in range(n_neighbours):

                random_location = random.randint(1, source_markers.shape[0] - 1)

                neighbour_change = random.random() > 0.95
                # is_correspondence_an_outlier = random.random() > 0.5

                if neighbour_change:
                    # Change location of source neighbour to add noise
                    source_neighbour[j, :] = source_markers[random_location, :]

                else:
                    source_neighbour[j, :] = source_markers[indices[i, j], :]

            # Perform SVD for the vector array
            # Select a random point that is at a significant distance, we do not want the code selecting the one right next to the original
            while distance_to_original_point < acceptable_distance:
                random_correspondence = random.randint(1, matches.shape[0] - 1)
                random_source = matches[random_correspondence, :2]
                distance_to_original_point = math.dist(random_source, matches[i, :2])

            distances_norm[i, :] = calculate_rotation_norm(matches[i, :2], source_neighbour,
                                                           matches[i, 2:], target_markers[indices[i, :5], :])
            outlier_norm[i, :] = calculate_rotation_norm(random_source, source_markers[indices[i, :5], :],
                                                         matches[i, 2:], target_markers[indices[i, :5], :])

        else:

            # For each of the points compute the SVD to match target
            distances_norm[i, :] = calculate_rotation_norm(matches[i, :2], source_markers[indices[i, :5], :],
                                                           matches[i, 2:], target_markers[indices[i, :5], :])
            # Select a random point that is at a significant distance, we do not want the code selecting the one right next to the original
            while distance_to_original_point < acceptable_distance:
                random_correspondence = random.randint(1, matches.shape[0] - 1)
                random_source = matches[random_correspondence, :2]
                distance_to_original_point = math.dist(random_source, matches[i, :2])

            # Create an outlier
            outlier_norm[i, :] = calculate_rotation_norm(random_source, source_markers[indices[i, :5], :],
                                                         matches[i, 2:], target_markers[indices[i, :5], :])

    # Descriptor difference
    descr_diff = np.linalg.norm(descriptors[:, :128] - descriptors[:, 128:], axis=1)
    # Add some random noise to the descriptors for outliers
    out_desc_diff = np.linalg.norm(descriptors[:, :128] - descriptors[:, 128:], axis=1) + np.max(descr_diff)

    # Normalize array
    # mean = np.mean(distances_norm, axis=0)
    # std = np.std(distances_norm, axis=0)
    # distances_norm = (distances_norm - mean) / std
    # outlier_norm = (outlier_norm - mean) / std

    return distances_norm, outlier_norm, descr_diff, out_desc_diff
