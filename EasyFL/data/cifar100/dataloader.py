import torch
import torchvision
import numpy as np
from torch.utils.data import Subset
import torchvision.transforms as transforms
import sys
import os
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../data")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))

from data.utils import direchlet_partition


def load_partition_data_cifar100(
    partition_alpha, client_number, batch_size
):
    CIFAR_MEAN = [0.485, 0.456, 0.406]
    CIFAR_STD = [0.229, 0.224, 0.225]

    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(32),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD) 
    ])

    trainset = torchvision.datasets.CIFAR100(root='../data/cifar100', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='../data/cifar100', train=False, download=True, transform=transform_test)
    train_data_num = len(trainset)
    test_data_num = len(testset)
    class_num = 100
    logging.info("*********partition data***************")
    file_idx_clients = direchlet_partition(trainset.targets, client_number, partition_alpha)
    data_local_num_dict = {}
    train_data_local_dict = {}
    test_data_local_dict = {}
    for i in range(client_number):
        data_local_num_dict[i] = len(file_idx_clients[i])
        train_data_local_dict[i] = torch.utils.data.DataLoader(Subset(trainset, file_idx_clients[i]), batch_size=batch_size, shuffle=True, num_workers=10, drop_last = True)
        logging.info("client id = %d, local_sample_number = %d" % (i, data_local_num_dict[i]))
    train_data_global = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=10)
    test_data_global = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=10)
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

# if __name__ == "__main__":
#     (
#     train_data_num,
#     test_data_num,
#     train_data_global,
#     test_data_global,
#     train_data_local_num_dict,
#     train_data_local_dict,
#     test_data_local_dict,
#     class_num,
#     ) = load_partition_data_cifar100(
#         0.1,
#         10,
#         32,
#     )
#     print(train_data_local_num_dict)