import torch
import torchvision
import numpy as np
from torch.utils.data import Subset
import sys
import os
import logging
import torchvision.transforms as transforms

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../data")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))

from data.utils import direchlet_partition


def load_partition_data_food101(
    partition_alpha, client_number, batch_size
):
    train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    food101_train = torchvision.datasets.Food101(
            root='../data/food101', 
            split='train', 
            transform=train_transform, 
            download=True
        )

    test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    food101_test = torchvision.datasets.Food101(
            root='../data/food101', 
            split='test', 
            transform=test_transform, 
            download=True
        )

    train_data_num = len(food101_train)
    test_data_num = len(food101_test)
    class_num = 101
    logging.info("*********partition data***************")
    file_idx_clients = direchlet_partition(food101_train._labels, client_number, partition_alpha)
    data_local_num_dict = {}
    train_data_local_dict = {}
    test_data_local_dict = {}
    for i in range(client_number):
        data_local_num_dict[i] = len(file_idx_clients[i])
        train_data_local_dict[i] = torch.utils.data.DataLoader(Subset(food101_train, file_idx_clients[i]), batch_size=batch_size, shuffle=True, num_workers=10)
        logging.info("client id = %d, local_sample_number = %d" % (i, data_local_num_dict[i]))
    train_data_global = torch.utils.data.DataLoader(food101_train, batch_size=batch_size, shuffle=True, num_workers=10)
    test_data_global = torch.utils.data.DataLoader(food101_test, batch_size=batch_size, shuffle=True, num_workers=10)
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
    ) = load_partition_data_food101(
        0.1,
        10,
        32,
    )

