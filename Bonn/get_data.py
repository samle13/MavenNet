import torch
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader
import os
import h5py
import random
import numpy as np


# 固定随机种子
SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_files = self.get_h5py_files(data_dir)
        self.class_mapping = {'Z': 0, 'N': 1, 'S': 2, 'O':0,'F':1}
        # self.class_mapping = {'Z': 0, 'F': 1, 'S':2}
        self.class_counts = [0] * len(self.class_mapping)
        self.count_class_samples()
        self.transform = transform  # 如果需要应用转换，请添加此参数

    def count_class_samples(self):
        for file_path in self.data_files:
            with h5py.File(file_path, 'r') as h5file:
                seizure_type = file_path.split('_')[-2]
                file_name = os.path.basename(file_path)

                seizure_type = file_name[0]

                label = self.class_mapping.get(seizure_type, -1)
                if label >= 0:
                    self.class_counts[label] += 1

    def get_h5py_files(self, root_dir):
        h5py_files = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.h5'):
                    h5py_files.append(os.path.join(root, file))
        return h5py_files

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        file_path = self.data_files[index]
        with h5py.File(file_path, 'r') as h5file:
            cwt_coeffs = h5file['cwt_coeffs'][:]
        
        file_name = os.path.basename(file_path)

        seizure_type = file_name[0]

        label = self.class_mapping.get(seizure_type, -1)

        # cwt_coeffs_normalized = self.normalize_data(cwt_coeffs)

        # if self.transform is not None:
        #     cwt_coeffs_normalized = self.transform(cwt_coeffs_normalized)

        return torch.tensor(cwt_coeffs, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def normalize_data(self, data):
        mean = data.mean(axis=(0, 1), keepdims=True)
        std = data.std(axis=(0, 1), keepdims=True)
        normalized_data = (data - mean) / std
        return normalized_data


def get_loaders(args):

    dataset = CustomDataset(args.data_dir, transform=None)
    print(dataset.class_counts)

    train_split = 0.8
    test_split = 0.2
    num_samples = len(dataset)
    train_size = int(num_samples * train_split)
    val_size = num_samples - train_size

    indices = list(range(num_samples))
    random.shuffle(indices)

    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

        # 创建数据集采样器
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    # 创建数据加载器
    batch_size = args.batch_size
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler,  num_workers= args.num_workers)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, num_workers= args.num_workers)

    args.num_class = len(dataset.class_mapping)
    return train_loader, test_loader