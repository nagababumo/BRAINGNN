import numpy as np
from sklearn.model_selection import StratifiedKFold
import pickle
import torch

def train_val_test_split(fold=0, n_splits=10):
    # Load labels from the pickle file
    with open('/teamspace/studios/this_studio/data/time-series_all_label.pkl', 'rb') as f:
        labels = pickle.load(f)
    
    # Convert labels to numpy array if needed
    labels = np.array(labels)
    
    # Ensure the labels length matches the dataset length
    assert len(labels) == 486, "Labels length does not match the dataset length"

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    splits = list(skf.split(np.zeros(len(labels)), labels))
    train_val_idx, test_idx = splits[fold]

    # Split train_val_idx into train_idx and val_idx
    train_idx, val_idx = list(StratifiedKFold(n_splits=2, shuffle=True, random_state=0).split(np.zeros(len(train_val_idx)), labels[train_val_idx]))[0]

    # Convert train_idx and val_idx to actual indices in the full dataset
    train_idx = train_val_idx[train_idx]
    val_idx = train_val_idx[val_idx]
    
    # Check for valid indices
    assert max(train_idx) < len(labels) and max(val_idx) < len(labels) and max(test_idx) < len(labels), "Index out of range"

    return train_idx, val_idx, test_idx

def fix_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    # Define the arguments
    n_epochs = 50
    batchSize = 50
    fold = 0
    lr = 0.01
    stepsize = 20
    gamma = 0.5
    weightdecay = 5e-3
    dataroot = '/teamspace/studios/this_studio/data'

    # Load dataset
    dataset = ABIDEDataset(dataroot, "time_series_all_label")

    # Perform train-validation-test split
    tr_index, val_index, te_index = train_val_test_split(fold=fold)

    # Print dataset length for debugging
    print(f"Dataset length: {len(dataset)}")
    
    # Debugging: Print indices to check
    print(f"Train indices: {tr_index}")
    print(f"Validation indices: {val_index}")
    print(f"Test indices: {te_index}")

    # Ensure the indices are valid for the dataset
    assert max(tr_index) < len(dataset) and max(val_index) < len(dataset) and max(te_index) < len(dataset), "Index out of range in dataset"

    # Convert indices to list if needed
    tr_index = tr_index.tolist() if isinstance(tr_index, np.ndarray) else tr_index
    val_index = val_index.tolist() if isinstance(val_index, np.ndarray) else val_index
    te_index = te_index.tolist() if isinstance(te_index, np.ndarray) else te_index

    # Select the datasets using the indices
    train_dataset = [dataset[i] for i in tr_index]
    val_dataset = [dataset[i] for i in val_index]
    test_dataset = [dataset[i] for i in te_index]

    # Debugging: Print lengths to check
    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Validation dataset length: {len(val_dataset)}")
    print(f"Test dataset length: {len(test_dataset)}")

    # Rest of your training code
    # ...
