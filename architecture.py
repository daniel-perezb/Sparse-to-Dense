import torch
import torch.nn as nn


# Define the model
class MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 200)
        self.fc2 = nn.Linear(200, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_classes)

        self.bn1 = nn.BatchNorm1d(200)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.001)

    def forward(self, x):
        # Layer 1
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Layer 2
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # Layer 3
        out = self.fc3(out)
        out = self.bn3(out)
        out = self.relu(out)
        # out = self.dropout(out)

        # Layer 4
        out = self.fc4(out)
        out = self.bn4(out)
        out = self.relu(out)

        # Output
        out = nn.functional.softmax(out, dim=1)

        return out

