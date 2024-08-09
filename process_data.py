import sys
import argparse
import numpy as np
import deepdish as dd
import warnings
import os

warnings.filterwarnings("ignore")
root_folder = '/teamspace/studios/this_studio/data'
data_folder = os.path.join(root_folder, 'processed_data', 'raw')

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def create_fully_connected_edge_index(num_nodes):
    row = np.repeat(np.arange(num_nodes), num_nodes)
    col = np.tile(np.arange(num_nodes), num_nodes)
    edge_index = np.vstack([row, col])
    edge_index = edge_index[:, row != col]  # Remove self-loops
    return edge_index

def load_data():
    time_series_data = np.load("/teamspace/studios/this_studio/data/processed_data/time_series_data.npz")
    labels = np.load("/teamspace/studios/this_studio/data/processed_data/labels.npy", allow_pickle=True)
    return time_series_data, labels

def main():
    parser = argparse.ArgumentParser(description='Process time-series data for BrainGNN.')
    parser.add_argument('--seed', default=123, type=int, help='Seed for random initialization.')
    parser.add_argument('--nclass', default=2, type=int, help='Number of classes.')
    args = parser.parse_args()

    time_series_data, labels = load_data()

    # Debugging: Print the loaded labels
    print(f"Loaded labels: {labels}")

    for i, key in enumerate(time_series_data.files):
        time_series = time_series_data[key]
        corr = np.corrcoef(time_series.T)  # Transpose to get correlations between regions
        pcorr = np.linalg.pinv(np.cov(time_series.T))  # Pseudo-inverse of the covariance matrix

        edge_index = create_fully_connected_edge_index(time_series.shape[1])

        # Ensure the label is being correctly indexed
        label = labels[i]
        print(f"Processing file {key}: Label = {label}")

        # Save the processed data
        data = {
            'corr': corr,
            'pcorr': pcorr,
            'label': label,
            'edge_index': edge_index
        }
        file_path = os.path.join(data_folder, f"{key}.h5")
        dd.io.save(file_path, data)
        print(f"Saved {file_path} with data keys: {list(data.keys())}")

    print("Processing complete.")

if __name__ == '__main__':
    main()
