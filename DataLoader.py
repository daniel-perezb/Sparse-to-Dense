from torch.utils.data import Dataset
import json
import torch
import numpy as np
import sparse_dense


class FeatureDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path

        # Read SIFT file
        with open(str(self.file_path), 'r') as f:
            data = json.load(f)
        self.descriptors = np.array(data['descriptors'])
        self.matches = np.array(data['points'])
        self.source_markers = np.array(data['source_markers'])
        self.target_markers = np.array(data['target_markers'])
        # Read image
        # self.source = cv2.imread(data['source_name'])
        # self.target = cv2.imread(data['target_name'])
        self.distances_norm, self.outlier_norm, self.desc_diff, self.out_desc_diff = \
            sparse_dense.descriptors_norm(self.descriptors, self.matches, self.source_markers, self.target_markers, data_augmentation=True)

        # Create data and labels variables
        self.norm_distances = np.concatenate([self.distances_norm, self.outlier_norm], axis=0)
        self.all_desc = np.concatenate([self.desc_diff, self.out_desc_diff], axis=0)
        self.data = np.column_stack((self.all_desc, self.norm_distances))
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.labels = np.concatenate((np.ones(int(self.data.shape[0]/2)), np.zeros(int(self.data.shape[0]/2))))
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        # get the length of the data
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# with open('fleece00/sift_02.json', 'r') as f:
#     data = json.load(f)
# descriptors = np.array(data['descriptors'])
# matches = np.array(data['points'])
# source_markers = np.array(data['source_markers'])
# target_markers = np.array(data['target_markers'])