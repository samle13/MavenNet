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
        self.seizure_mapping = {'TCSZ': 0, 'FNSZ': 1, 'CPSZ': 2, 'GNSZ': 3, 'SPSZ': 4, 'TNSZ': 5, 'ABSZ': 6, 'MYSZ': 7}
        self.patient_mapping = {
            '00000258': 0,
            '00001027': 1,
            '00001984': 2,
            '00005479': 3,
            '00006544': 4,
            '00006546': 5,
            '00008174': 6,
            '00008512': 7,
            '00008544': 8,
            '00008616': 9,
            '00008889': 10,
            '00009578': 11
        }
        self.patient_to_seizure = {
            '00000258': 'TCSZ',
            '00001027': 'CPSZ',
            '00001984': 'ABSZ',
            '00005479': 'CPSZ',
            '00006544': 'MYSZ',
            '00006546': 'GNSZ',
            '00008174': 'GNSZ',
            '00008512': 'FNSZ',
            '00008544': 'FNSZ',
            '00008616': 'SPSZ',
            '00008889': 'TNSZ',
            '00009578': 'TCSZ',
        }
        # self.class_counts = [0] * len(self.class_mapping)
        # self.count_class_samples()
        self.transform = transform  # 如果需要应用转换，请添加此参数
        patients = list(set(self.patient_to_seizure.keys()))
        seizures = list(set(self.patient_to_seizure.values()))

        # 创建矩阵并初始化为0
        matrix = np.zeros((len(patients), len(seizures)), dtype=int)

        # 根据字典设置矩阵元素为1
        for i, patient in enumerate(patients):
            seizure = self.patient_to_seizure[patient]
            j = seizures.index(seizure)
            matrix[i, j] = 1

        self.patient_seizure_matrix = matrix


    def count_class_samples(self):
        for file_path in self.data_files:
            with h5py.File(file_path, 'r') as h5file:
                seizure_type = file_path.split('_')[-2]
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
        seizure_type = file_path.split('_')[-2]
        patient_id = file_path.split('_')[3]

        label = self.seizure_mapping.get(seizure_type, -1)
        label_patient = self.patient_mapping.get(patient_id, -1)


        cwt_coeffs_normalized = self.normalize_data(cwt_coeffs)

        if self.transform is not None:
            cwt_coeffs_normalized = self.transform(cwt_coeffs_normalized)

        return torch.tensor(cwt_coeffs_normalized, dtype=torch.float32), torch.tensor(label,
                                                                                      dtype=torch.long), torch.tensor(
            label_patient, dtype=torch.long)

    def normalize_data(self, data):
        mean = data.mean(axis=(0, 2), keepdims=True)
        std = data.std(axis=(0, 2), keepdims=True)
        normalized_data = (data - mean) / std
        return normalized_data


def get_loaders(args):
    dataset_train = CustomDataset(args.train_dir, transform=None)
    dataset_test = CustomDataset(args.test_dir, transform=None)
    # train_split = 0.8
    # test_split = 0.2
    # num_samples = len(dataset)
    # train_size = int(num_samples * train_split)
    # val_size = num_samples - train_size
    #
    # indices = list(range(num_samples))
    # random.shuffle(indices)
    #
    # train_indices = indices[:train_size]
    # test_indices = indices[train_size:]
    #
    # 创建数据集采样器
    # train_sampler = SubsetRandomSampler(train_indices)
    # test_sampler = SubsetRandomSampler(test_indices)

    # 创建数据加载器
    batch_size = args.batch_size
    train_loader = DataLoader(dataset_train, batch_size=batch_size, num_workers=args.num_workers,shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, num_workers=args.num_workers)

    args.num_class = 12
    return train_loader, test_loader, dataset_train.patient_seizure_matrix
