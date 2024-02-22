import torch
import torchvision
import numpy as np
from torch.utils.data import Subset
import sys
import os
import logging
import torchvision.transforms as transforms
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../data")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))

from data.utils import direchlet_partition
def train_test_split(labels_idx, class_num, test_ratio):
    train_idx = []
    test_idx = []
    for i in range(class_num):
        class_idx = list(np.where(np.array(labels_idx) == i)[0])
        test_class_num = round(len(class_idx) * test_ratio)
        test_idx_class = random.sample(class_idx, test_class_num)
        train_idx_class = [idx for idx in class_idx if idx not in test_idx_class]
        train_idx.extend(train_idx_class)
        test_idx.extend(test_idx_class)
    return train_idx, test_idx

def load_partition_data_eurosat(
    partition_alpha, client_number, batch_size
):
    train_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.RandomRotation(90),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.Resize(256),
                                    transforms.RandomCrop(224)])
    eurosat_train = torchvision.datasets.EuroSAT(root='../data/eurosat', download=True, transform=train_transform)
    class_num = 10
    test_ratio = 0.2
    train_idx, test_idx = train_test_split(eurosat_train.targets, class_num, test_ratio)
    train_data_num = len(train_idx)
    test_data_num = len(test_idx)
    train_idx_labels = [eurosat_train.targets[i] for i in train_idx]

    logging.info("*********partition data***************")
    file_idx_clients = direchlet_partition(train_idx_labels, client_number, partition_alpha)
    data_local_num_dict = {}
    train_data_local_dict = {}
    test_data_local_dict = {}
    for i in range(client_number):
        data_local_num_dict[i] = len(file_idx_clients[i])
        client_idx = [train_idx[i] for i in file_idx_clients[i]]
        train_data_local_dict[i] = torch.utils.data.DataLoader(Subset(eurosat_train, client_idx), batch_size=batch_size, shuffle=True, num_workers=10)
        logging.info("client id = %d, local_sample_number = %d" % (i, data_local_num_dict[i]))
    train_data_global = torch.utils.data.DataLoader(eurosat_train, batch_size=batch_size, shuffle=True, num_workers=10)

    test_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224)])
    eurosat_test = torchvision.datasets.EuroSAT(root='../data/eurosat', download=True, transform=test_transform)
    test_data_global = torch.utils.data.DataLoader(Subset(eurosat_test, test_idx), batch_size=batch_size, shuffle=True, num_workers=10)
    logging.info("##########test sample number = %d" % (len(Subset(eurosat_test, test_idx))))
    return (
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    )

if __name__ == "__main__":
    (
    train_data_num,
    test_data_num,
    train_data_global,
    test_data_global,
    train_data_local_num_dict,
    train_data_local_dict,
    test_data_local_dict,
    class_num,
    ) = load_partition_data_eurosat(
        0.1,
        10,
        32,
    )