from torch import nn
import torch.nn.functional as F


class TurbofanModel(nn.Module):
    def __init__(self, train_mean, train_std, input_size):
        super().__init__()

        self.train_mean = train_mean
        self.train_std = train_std

        self.fc1 = nn.Linear(input_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, 24)
        self.fc4 = nn.Linear(24, 1)

    # extend send function to also send the mean and std params
    def send(self, worker):
        super().send(worker)
        self.train_mean = self.train_mean.send(worker)
        self.train_std = self.train_std.send(worker)

    # extend get function to also get the mean and std params
    def get(self):
        super().get()
        self.train_mean = self.train_mean.get()
        self.train_std = self.train_std.get()

    def forward(self, x):
        # scale the input
        x = (x - self.train_mean) / self.train_std

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = x[:, -1, :]

        return x


class BatchNormFCModel(nn.Module,):
    def __init__(self, window_size, features_size):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(window_size)
        self.fc1 = nn.Linear(features_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, 24)
        self.fc4 = nn.Linear(24, 1)

    # extend send function to also send the mean and std params
    def send(self, worker):
        super().send(worker)

    # extend get function to also get the mean and std params
    def get(self):
        super().get()

    def forward(self, x):
        x = F.relu(self.bn1(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = x[:, -1, :]

        return x
