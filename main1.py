import os
import torch
import numpy as np
from torch_geometric.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from torch.optim import lr_scheduler
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import copy
import time
import argparse
from collections.abc import Mapping
# Import necessary modules from your custom files
from imports.ABIDEDataset import ABIDEDataset
from net.braingnn import Network
from imports.utils import fix_seed, train_val_test_split

# Define your options here or parse from command line
parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=50, help='number of epochs')
parser.add_argument('--batchSize', type=int, default=50, help='batch size')
parser.add_argument('--fold', type=int, default=0, help='fold index')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--stepsize', type=int, default=20, help='scheduler step size')
parser.add_argument('--gamma', type=float, default=0.5, help='scheduler gamma')
parser.add_argument('--weightdecay', type=float, default=5e-3, help='weight decay')
parser.add_argument('--dataroot', type=str, default='/teamspace/studios/this_studio/data', help='data root path')
parser.add_argument('--save_path', type=str, default='./models/', help='model save path')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--lamb0', type=float, default=1.0, help='lambda0')
parser.add_argument('--lamb1', type=float, default=1.0, help='lambda1')
parser.add_argument('--lamb2', type=float, default=1.0, help='lambda2')
parser.add_argument('--lamb3', type=float, default=1.0, help='lambda3')
parser.add_argument('--lamb4', type=float, default=1.0, help='lambda4')
parser.add_argument('--lamb5', type=float, default=1.0, help='lambda5')
parser.add_argument('--ratio', type=float, default=0.5, help='pooling ratio')
parser.add_argument('--nclass', type=int, default=2, help='number of classes')
parser.add_argument('--indim', type=int, default=116, help='input dimension')

opt = parser.parse_args()

# Set random seed for reproducibility
fix_seed(opt.seed)

# Initialize tensorboard writer
writer = SummaryWriter()

# Check the contents of the data directory
print(f"Data directory contents: {os.listdir(opt.dataroot)}")

# Load dataset
dataset = ABIDEDataset(root=opt.dataroot)

# Print the length of the dataset for debugging
print(f"Dataset loaded with length: {len(dataset)}")

# Print the first few elements of the dataset for inspection
for i in range(min(5, len(dataset))):
    print(f"Dataset element {i}: {dataset[i]}")

# Ensure the dataset length is correct
assert len(dataset) > 2, "The dataset appears to be loaded incorrectly."

# Split dataset into train, validation, and test sets using StratifiedKFold
train_idx, val_idx, test_idx = train_val_test_split(fold=opt.fold)

# Print dataset split lengths for debugging
print(f"Train indices length: {len(train_idx)}, Validation indices length: {len(val_idx)}, Test indices length: {len(test_idx)}")

# Print the maximum index in each split to check if they are within bounds
print(f"Max train index: {max(train_idx)}, Max val index: {max(val_idx)}, Max test index: {max(test_idx)}")

# Ensure indices are within the dataset length
assert max(train_idx) < len(dataset) and max(val_idx) < len(dataset) and max(test_idx) < len(dataset), "Index out of range in dataset"
# Convert indices to lists of native Python integers
train_idx = train_idx.tolist()
val_idx = val_idx.tolist()
test_idx = test_idx.tolist()

# Create data loaders
train_dataset = [dataset[i] for i in train_idx]
val_dataset = [dataset[i] for i in val_idx]
test_dataset = [dataset[i] for i in test_idx]

train_loader = DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=opt.batchSize, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=False)


# Define the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Network(opt.indim, opt.ratio, opt.nclass).to(device)
print(model)

# Define the optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weightdecay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.stepsize, gamma=opt.gamma)

# Define loss functions
EPS = 1e-10

def topk_loss(s, ratio):
    if ratio > 0.5:
        ratio = 1 - ratio
    if s is None:
        return torch.tensor(0.0, device='cuda')  # Or another appropriate default value

    print(f"Shape of s: {s.shape}")

    # Handle different shapes of s
    if len(s.shape) == 1:
        s = s.sort().values
        res = -torch.log(s[-int(s.size(0) * ratio):] + EPS).mean() - torch.log(1 - s[:int(s.size(0) * ratio)] + EPS).mean()
    elif len(s.shape) == 2:
        s = s.sort(dim=1).values
        res = -torch.log(s[:, -int(s.size(1) * ratio):] + EPS).mean() - torch.log(1 - s[:, :int(s.size(1) * ratio)] + EPS).mean()
    else:
        raise ValueError(f"Unsupported shape for s: {s.shape}")

    return res


def consist_loss(s):
    if len(s.shape) == 1:
        s = s.unsqueeze(1)  # Add an extra dimension to make it 2D
    elif len(s.shape) != 2:
        raise ValueError(f"Unsupported shape for s in consist_loss: {s.shape}")

    s = torch.sigmoid(s)
    W = torch.ones(s.shape[0], s.shape[0], device=s.device)
    D = torch.eye(s.shape[0], device=s.device) * torch.sum(W, dim=1)
    L = D - W

    res = torch.trace(torch.transpose(s, 0, 1) @ L @ s) / (s.shape[0] * s.shape[0])
    return res

# Training function
def train(epoch):
    model.train()
    loss_all = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out, s, w = model(data.x, data.edge_index, data.batch)
        
        if s is None:
            s = torch.zeros_like(data.x)  # Handle the case where s is None
        print(f"Shape of s before loss computation: {s.shape}")
        loss_cls = F.nll_loss(out, data.y)
        loss_s = topk_loss(s, opt.lamb1)
        loss_w = consist_loss(w)
        loss = opt.lamb0 * loss_cls + opt.lamb2 * loss_s + opt.lamb3 * loss_w
        loss.backward()
        optimizer.step()
        loss_all += loss.item() * data.num_graphs

    scheduler.step()
    return loss_all / len(train_loader.dataset), loss_cls.item(), loss_s.item(), loss_w.item()



# Evaluation function
def test(loader):
    model.eval()
    correct = 0

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out, _, _ = model(data.x, data.edge_index, data.batch)
            pred = out.max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()

    return correct / len(loader.dataset)

# Test loss function
def test_loss(loader, epoch):
    model.eval()
    loss_all = 0

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out, s, w = model(data.x, data.edge_index, data.batch)
            loss_cls = F.nll_loss(out, data.y)
            loss_s = topk_loss(s, opt.lamb1)
            loss_w = consist_loss(w)
            loss = opt.lamb0 * loss_cls + opt.lamb2 * loss_s + opt.lamb3 * loss_w

            loss_all += loss.item() * data.num_graphs

    return loss_all / len(loader.dataset)

# Run training and validation
if __name__ == "__main__":
    for epoch in range(opt.n_epochs):
        loss, loss_cls, loss_s, loss_w = train(epoch)
        train_acc = test(train_loader)
        val_acc = test(val_loader)
        test_l = test_loss(val_loader, epoch)

        print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Loss_cls: {loss_cls:.4f}, Loss_s: {loss_s:.4f}, Loss_w: {loss_w:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Loss: {test_l:.4f}")
        writer.add_scalar('data/train_acc', train_acc, epoch)
        writer.add_scalar('data/val_acc', val_acc, epoch)
        writer.add_scalar('data/test_loss', test_l, epoch)
        writer.add_scalar('loss/total', loss, epoch)
        writer.add_scalar('loss/classification', loss_cls, epoch)
        writer.add_scalar('loss/topk', loss_s, epoch)
        writer.add_scalar('loss/consistency', loss_w, epoch)

    writer.close()
