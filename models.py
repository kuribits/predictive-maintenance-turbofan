from torch import nn
import torch.nn.functional as F
import syft.frameworks.torch.nn.rnn as syftnn

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


class VanillaGRU(nn.Module):
    def __init__(self, window_size, features_size):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(window_size)
        self.lstm = syftnn.GRU(input_size=features_size, hidden_size=1, batch_first=True)
        self.fc1 = nn.Linear(80, 1)

    def forward(self, x):
        x = F.relu(self.bn1(x))
        x = F.relu(self.lstm(x)[0])
        x = x.permute(0,2,1)
        x = self.fc1(x)
        x = x[:, -1, :]

        return x


class VanillaLSTM(nn.Module):
    def __init__(self, window_size, features_size):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(window_size)
        self.lstm = syftnn.LSTM(input_size=features_size, hidden_size=1, batch_first=True)
        self.fc1 = nn.Linear(80, 1)

    def forward(self, x):
        x = F.relu(self.bn1(x))
        x = F.relu(self.lstm(x)[0])
        x = x.permute(0,2,1)
        x = self.fc1(x)
        x = x[:, -1, :]

        return x


class BatchNormFCModel(nn.Module):
    def __init__(self, window_size, features_size):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(window_size)
        self.fc1 = nn.Linear(features_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, 24)
        self.fc4 = nn.Linear(24, 1)

    def forward(self, x):
        x = F.relu(self.bn1(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = x[:, -1, :]

        return x
