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


def load_partition_data_flower102(
    partition_alpha, client_number, batch_size
):
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        # transforms.RandomRotation(45),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    trainset = torchvision.datasets.Flowers102(root='../data/flower102', split = 'train', transform = train_transform, download = True)
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    testset = torchvision.datasets.Flowers102(root='../data/flower102', split = 'test', transform = test_transform, download = True)
    train_data_num = len(trainset)
    test_data_num = len(testset)
    class_num = 102
    logging.info("*********partition data***************")
    file_idx_clients = direchlet_partition(trainset._labels, client_number, partition_alpha)
    data_local_num_dict = {}
    train_data_local_dict = {}
    test_data_local_dict = {}
    for i in range(client_number):
        data_local_num_dict[i] = len(file_idx_clients[i])
        train_data_local_dict[i] = torch.utils.data.DataLoader(Subset(trainset, file_idx_clients[i]), batch_size=batch_size, shuffle=True, num_workers=2)
        logging.info("client id = %d, local_sample_number = %d" % (i, data_local_num_dict[i]))
    train_data_global = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_data_global = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)
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
    ) = load_partition_data_flower102(
        0.1,
        10,
        32,
    )