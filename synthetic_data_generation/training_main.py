import os
import sys
import torch
from torch.utils.data import Dataset
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from torchvision.models import vgg19
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from sklearn.metrics import f1_score
from tqdm import tqdm
import copy
import torchvision
import wandb
import numpy as np
import random
import matplotlib.pyplot as plt

from syn_dataloader import SynCifar10Dataset, SynFlower102Dataset, CIFAR100GeneratedDataset
from utils import progress_bar, direchlet_partition
from resnet import get_resnet
from resnet_pytorch import Resnet
from convnet import ConvNet

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

def data_loading(dataset, batch_size):
    if dataset == 'cifar10_real':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        print("############loading dataset ", dataset, "################")
        trainset = torchvision.datasets.CIFAR10(
            root='syn_dataset/cifar_test/cifar10_test_dataset', train=True, download=True, transform=transform_train)
        train_data_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=False, num_workers=6)
        testset = torchvision.datasets.CIFAR10(
            root='syn_dataset/cifar_test/cifar10_test_dataset', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=6)
        class_num = 10

    if dataset == 'cifar10_all_together':
        # lr = 1e-4
        print("############loading dataset ", dataset, "################")
        image_paths = "syn_dataset/cifar10_all_together"
        transform = transforms.Compose([
                                transforms.RandomResizedCrop(32),
                                transforms.RandomHorizontalFlip(),
                                # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        cifar10_set = SynCifar10Dataset(image_paths, transform=transform)
        train_data_loader = torch.utils.data.DataLoader(cifar10_set, batch_size=batch_size, shuffle=True, num_workers=6)
        test_transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize(32),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        cifar10_test = torchvision.datasets.CIFAR10(root='syn_dataset/cifar_test/cifar10_test_dataset', train=False, download=True, transform=test_transform)
        testloader = torch.utils.data.DataLoader(cifar10_test, batch_size=batch_size, shuffle=True, num_workers=6)
        class_num = 10
        
    if dataset == 'cifar100_real':
        print("############loading dataset ", dataset, "################")
        CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
        transform = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
        ])
        cifar100_set = torchvision.datasets.CIFAR100(root='syn_dataset/cifar_test/cifar10_test_dataset', train=True, download=True, transform=transform)
        train_data_loader = torch.utils.data.DataLoader(cifar100_set, batch_size=batch_size, shuffle=True, num_workers=16)
        test_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize(32),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        cifar100_test = torchvision.datasets.CIFAR100(root='syn_dataset/cifar_test/cifar10_test_dataset', train=False, download=True, transform=test_transform)
        testloader = torch.utils.data.DataLoader(cifar100_test, batch_size=batch_size, shuffle=True, num_workers=16)
        class_num = 100

    if dataset == 'cifar100_together':
        print("############loading dataset ", dataset, "################")
        image_paths = "syn_dataset/cifar100_generated_32A"
        transform = transforms.Compose([
                                    transforms.RandomResizedCrop(32),
                                    transforms.RandomHorizontalFlip(),
                                    # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        cifar100_set = CIFAR100GeneratedDataset(root=image_paths, transforms=transform)
        train_data_loader = torch.utils.data.DataLoader(cifar100_set, batch_size=batch_size, shuffle=True, num_workers=16)
        test_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize(32),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        cifar100_test = torchvision.datasets.CIFAR100(root='syn_dataset/cifar_test/cifar10_test_dataset', train=False, download=True, transform=test_transform)
        testloader = torch.utils.data.DataLoader(cifar100_test, batch_size=batch_size, shuffle=True, num_workers=16)
        class_num = 100

    if dataset == 'flower102_real':
        print("############loading dataset ", dataset, "################")
        image_paths = "syn_dataset/flower102/syn_dataset"
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            # transforms.RandomRotation(45),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        flower_train = torchvision.datasets.Flowers102(root='syn_dataset/flower102/test_dataset', split='train', transform=train_transform, download=True)
        train_data_loader = torch.utils.data.DataLoader(flower_train, batch_size=batch_size, shuffle=True, num_workers=16)
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        flower_test = torchvision.datasets.Flowers102(root='syn_dataset/flower102/test_dataset', split='test', transform=test_transform, download=True)
        testloader = torch.utils.data.DataLoader(flower_test, batch_size=batch_size, shuffle=True, num_workers=16)
        class_num = 102

    if dataset == 'flower102_syn':
        print("############loading dataset ", dataset, "################")
        image_paths = "syn_dataset/flower102/syn_dataset"
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            # transforms.RandomRotation(45),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        flower_train = SynFlower102Dataset(data_dir=image_paths, transform=train_transform)
        train_data_loader = torch.utils.data.DataLoader(flower_train, batch_size=batch_size, shuffle=True, num_workers=16)
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        flower_test = torchvision.datasets.Flowers102(root='syn_dataset/flower102/test_dataset', split='test', transform=test_transform, download=True)
        testloader = torch.utils.data.DataLoader(flower_test, batch_size=batch_size, shuffle=True, num_workers=16)
        class_num = 102

    return train_data_loader, testloader, class_num


def model_device(model, device_idx, class_num):
    if model == 'resnet20':
        model = get_resnet('resnet20')(class_num, False)
    if model == 'resnet18':
        model = Resnet(
            num_classes=class_num, resnet_size=18, pretrained=False)
    if model == 'resnet50':
        model = Resnet(
            num_classes=class_num, resnet_size=50, pretrained=False)
    if model == 'vgg19':
        model = vgg19(pretrained = True)
        input_lastLayer = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(input_lastLayer,class_num)
    if model == 'convnet':
        model = ConvNet(num_classes=class_num)
    device = "cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu"
    return model, device

def train_test(train_data, test_data, model, device, epochs, learning_rate, model_save_path):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    # 0.3 weight decay for eurosat syn data
    # 0.9 weight decay for cifar syn data
    # real image use SGD, fake image use AdamW
    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.9)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                        momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    epoch_loss = []
    for epoch in range(epochs):
        print('\nEpoch: %d' % epoch)
        model.train()  
        train_loss = 0
        correct = 0
        total = 0
        #  data, labels, lens
        for batch_idx, (inputs, targets) in enumerate(train_data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(train_data_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        wandb.log({"Train/Acc": 100.*correct/total, "epoch": epoch})
        wandb.log({"Train/Loss": train_loss/(batch_idx+1), "epoch": epoch})
        
        if (epoch % 1 == 0):
            print("################testing#######################")
            model.eval()        
            metrics = {"test_correct": 0, "test_loss": 0, "test_total": 0, "test_f1": 0}
            with torch.no_grad():
                label_list, pred_list = list(), list()
                for batch_idx, (data, labels) in enumerate(tqdm(testloader)):
                    # for data, labels, lens in test_data:
                    data, labels = data.to(device), labels.to(device)
                    output = model(data)
                    loss = criterion(output, labels).data.item()
                    pred = output.data.max(1, keepdim=True)[
                        1
                    ]  # get the index of the max log-probability
                    correct = pred.eq(labels.data.view_as(pred)).sum()
                    for idx in range(len(labels)):
                        label_list.append(labels.detach().cpu().numpy()[idx])
                        pred_list.append(pred.detach().cpu().numpy()[idx][0])

                    metrics["test_correct"] += correct.item()
                    metrics["test_loss"] += loss * labels.size(0)
                    metrics["test_total"] += labels.size(0)
            metrics["test_f1"] = f1_score(label_list, pred_list, average='macro')
            test_num_samples = []
            test_tot_corrects = []
            test_losses = []
            test_f1s = []
            test_tot_correct, test_num_sample, test_loss, test_f1 = (
                metrics["test_correct"],
                metrics["test_total"],
                metrics["test_loss"],
                metrics["test_f1"],
            )
            test_tot_corrects.append(copy.deepcopy(test_tot_correct))
            test_num_samples.append(copy.deepcopy(test_num_sample))
            test_losses.append(copy.deepcopy(test_loss))
            test_f1s.append(copy.deepcopy(test_f1))
            # test on test dataset
            test_acc = sum(test_tot_corrects) / sum(test_num_samples)
            test_loss = sum(test_losses) / sum(test_num_samples)
            wandb.log({"Test/Sample Number": sum(test_num_samples), "epoch": epoch})
            wandb.log({"Test/Acc": test_acc, "epoch": epoch})
            wandb.log({"Test/F1": test_f1, "epoch": epoch})
            wandb.log({"Test/Loss": test_loss, "epoch": epoch})
            stats = {"test_acc": test_acc, 
                    "test_f1": test_f1, 
                    "test_loss": test_loss}
            print(stats)
        scheduler.step()
    torch.save(model.state_dict(), model_save_path)

if __name__ == "__main__":

    data = 'cifar100'
    model = 'resnet50'
    batch_size = 32
    epochs = 200
    lr = 1e-4
    device = 7
    model_idx = 'pretrain_cl_1'
    save_path = 'model_check_point/' + data + "_" + model + "_" + str(model_idx) + ".pth"
    wandb.init(
        mode = 'disabled',
        project = 'fediffusion',
        entity = 'ultraz',
        name = data + '_res_sgd_cl_' + str(model_idx) + '_' + str(lr),
    )

    train_data_loader, testloader, class_num = data_loading(data, batch_size)
    model, device = model_device(model, device, class_num)
    train_test(train_data_loader, testloader, model, device, epochs, lr, model_idx)

 