import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, TopKPooling

class Network(nn.Module):
    def __init__(self, indim, ratio, nclass):
        super(Network, self).__init__()
        self.conv1 = GCNConv(indim, 64)
        self.pool1 = TopKPooling(64, ratio=ratio)
        self.conv2 = GCNConv(64, 64)
        self.pool2 = TopKPooling(64, ratio=ratio)

        self.lin1 = nn.Linear(64*2, 128)
        self.lin2 = nn.Linear(128, nclass)

    def forward(self, x, edge_index, batch, edge_attr=None, pos=None):
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, perm, score1 = self.pool1(x, edge_index, batch=batch)
        x1 = global_mean_pool(x, batch)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, perm, score2 = self.pool2(x, edge_index, batch=batch)
        x2 = global_mean_pool(x, batch)

        x = torch.cat([x1, x2], dim=1)

        x = F.relu(self.lin1(x))
        x = F.log_softmax(self.lin2(x), dim=-1)

        return x, score1, score2
