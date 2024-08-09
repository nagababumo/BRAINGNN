import os
import pickle
import deepdish as dd
import torch
from torch_geometric.data import InMemoryDataset, Data

class ABIDEDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        # Adjust the path to match where your labels file is actually located
        self.labels_file = os.path.join(root, 'processed_data', 'time-series_all_label.pkl')
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # Adjust to reflect the actual directory containing raw data
        return [f for f in os.listdir(self.raw_dir) if f.endswith('.h5')]

    @property
    def processed_file_names(self):
        return ['data.pt']

    @property
    def raw_dir(self):
        # Override this to point to the correct directory
        return os.path.join(self.root, 'processed_data', 'raw')

    def process(self):
        data_list = []

        with open(self.labels_file, 'rb') as f:
            labels = pickle.load(f)

        for i, raw_path in enumerate(self.raw_paths):
            data_dict = dd.io.load(raw_path)
            if all(key in data_dict for key in ['corr', 'pcorr', 'edge_index']):
                label = labels[i]
                data = Data(
                    x=torch.tensor(data_dict['corr'], dtype=torch.float),
                    edge_index=torch.tensor(data_dict['edge_index'], dtype=torch.long),
                    y=torch.tensor([label], dtype=torch.long)
                )
                data_list.append(data)
            else:
                print(f"Skipping file {os.path.basename(raw_path)}: Missing required data keys")

        if len(data_list) == 0:
            raise ValueError("No valid data was added to the data list. Check your file paths and content.")
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

if __name__ == "__main__":
    dataset = ABIDEDataset(root='/teamspace/studios/this_studio/data')
    print(f"Processed dataset: {dataset}")
