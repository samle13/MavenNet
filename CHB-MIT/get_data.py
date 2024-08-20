import torch
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader
import os
import h5py
import random
import numpy as np



def set_seed(seed):
    # fix seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.data_files = os.listdir(data_dir)
        self.class_mapping = {'0': 0, '1': 1}
        self.class_counts = [0] * len(self.class_mapping)
        self.count_class_samples()
        self.transform = transform  # 如果需要应用转换，请添加此参数

    def count_class_samples(self):
        for file in self.data_files:
            file_path = os.path.join(self.data_dir, file)
            with h5py.File(file_path, 'r') as h5file:
                label = file_path.split('_')[-1].split('.')[0]
                label = self.class_mapping.get(label, -1)
                if label >= 0:
                    self.class_counts[label] += 1

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        file = self.data_files[index]
        file_path = os.path.join(self.data_dir, file)
        with h5py.File(file_path, 'r') as h5file:
            cwt_coeffs = h5file['cwt_coeffs'][:]
            label = file_path.split('_')[-1].split('.')[0]
            label = self.class_mapping.get(label, -1)

        cwt_coeffs_normalized = self.normalize_data(cwt_coeffs)

        if self.transform is not None:
            cwt_coeffs_normalized = self.transform(cwt_coeffs_normalized)

        return torch.tensor(cwt_coeffs_normalized, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def normalize_data(self, data):
        mean = data.mean(axis=(0, 2), keepdims=True)
        std = data.std(axis=(0, 2), keepdims=True)
        normalized_data = (data - mean) / std
        return normalized_data


def get_loaders(args):
    set_seed(args.seed)

    dataset = CustomDataset(args.data_dir, transform=None)

    print(dataset.class_counts)

    train_split = 0.8
    test_split = 0.2
    num_samples = len(dataset)
    train_size = int(num_samples * train_split)

    indices = list(range(num_samples))
    random.shuffle(indices)

    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    # 创建数据集采样器
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    # 创建数据加载器
    batch_size = args.batch_size
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=args.num_workers)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, num_workers=args.num_workers)

    args.class_counts = dataset.class_counts
    args.num_class = len(dataset.class_mapping)
    return train_loader, test_loader