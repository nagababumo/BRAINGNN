import os
import pickle
import argparse
import numpy as np

# Input data variables
root_folder = '/teamspace/studios/this_studio/data'
data_folder = os.path.join(root_folder, 'processed_data/')
if not os.path.exists(data_folder):
    os.makedirs(data_folder)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def load_data(time_series_path, labels_path):
    with open(time_series_path, 'rb') as f:
        time_series_data = pickle.load(f)

    with open(labels_path, 'rb') as f:
        labels = pickle.load(f)

    return time_series_data, labels

def main():
    parser = argparse.ArgumentParser(description='Load time-series data and labels.')
    parser.add_argument('--time_series_path', default=os.path.join(root_folder, 'time_series_aal.pkl'),
                        type=str, help='Path to the time-series data file.')
    parser.add_argument('--labels_path', default=os.path.join(root_folder, 'time_series_all_label.pkl'),
                        type=str, help='Path to the labels data file.')
    args = parser.parse_args()
    print('Arguments: \n', args)

    try:
        time_series_data, labels = load_data(args.time_series_path, args.labels_path)
        print(f"Loaded time-series data of type: {type(time_series_data)}")
        print(f"Loaded labels of type: {type(labels)}")

        # Check contents
        print(f"Time series data: {time_series_data}")
        print(f"Labels: {labels}")

        # Save the loaded data in the data_folder
        np.savez(os.path.join(data_folder, 'time_series_data.npz'), *time_series_data)
        np.save(os.path.join(data_folder, 'labels.npy'), labels)
        print("Files saved successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
