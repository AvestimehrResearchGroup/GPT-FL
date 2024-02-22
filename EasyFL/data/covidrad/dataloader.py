from PIL import Image
import torchvision
import numpy as np
import shutil
import random
import torch
import os


import sys
import logging
import torchvision.transforms as transforms
from torch.utils.data import Subset

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../data")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))

from data.utils import direchlet_partition

class ChestXRayDataset(torch.utils.data.Dataset):
    def __init__(self, image_dirs, transform):
        def get_images(class_name): #Verificar se o aquivo realmente é imagem 'png'
            images = [x for x in os.listdir(image_dirs[class_name]) if x.lower().endswith('png')] 
            print(f'Found {len(images)} {class_name} examples')
            return images
        
        self.images = {}
        self.targets = []
        self.class_names = ['NORMAL', 'Viral Pneumonia', 'COVID']
        
        for class_name in self.class_names:
            class_images = get_images(class_name)
            self.images[class_name] = class_images
            self.targets.extend([self.class_names.index(class_name)] * len(class_images))
            
        self.image_dirs = image_dirs
        self.transform = transform
          
    def __len__(self):
        return sum([len(self.images[c]) for c in self.class_names])
    
    def __getitem__(self, index):
        class_name = random.choice(self.class_names)
        index = index%len(self.images[class_name])
        image_name = self.images[class_name][index]
        image_path = os.path.join(self.image_dirs[class_name], image_name)
        image = Image.open(image_path).convert('RGB')
        return self.transform(image), self.class_names.index(class_name)

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

def load_partition_data_covidrad(
    partition_alpha, client_number, batch_size
):  
    data_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size = (224, 224)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])
    train_dirs = {
    'NORMAL': '/home/ultraz/covid19rad/COVID-19_Radiography_Dataset/Normal/images',
    'Viral Pneumonia': '/home/ultraz/covid19rad/COVID-19_Radiography_Dataset/Viral Pneumonia/images',
    'COVID': '/home/ultraz/covid19rad/COVID-19_Radiography_Dataset/COVID/images'
    }

    total_dataset = ChestXRayDataset(train_dirs, data_transform)
    class_num = 3
    test_ratio = 0.2
    # total_labels = [label for _, label in total_dataset]
    total_labels = total_dataset.targets
    train_idx, test_idx = train_test_split(total_labels, class_num, test_ratio)
    train_data_num = len(train_idx)
    test_data_num = len(test_idx)
    train_idx_labels = [total_dataset.targets[i] for i in train_idx]
    logging.info("*********partition data***************")
    file_idx_clients = direchlet_partition(train_idx_labels, client_number, partition_alpha)
    data_local_num_dict = {}
    train_data_local_dict = {}
    test_data_local_dict = {}
    for i in range(client_number):
        data_local_num_dict[i] = len(file_idx_clients[i])
        client_idx = [train_idx[i] for i in file_idx_clients[i]]
        train_data_local_dict[i] = torch.utils.data.DataLoader(Subset(total_dataset, client_idx), batch_size=batch_size, shuffle=True, num_workers=10)
        logging.info("client id = %d, local_sample_number = %d" % (i, data_local_num_dict[i]))
    train_data_global = torch.utils.data.DataLoader(total_dataset, batch_size=batch_size, shuffle=True, num_workers=10)

    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size = (224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    total_dataset = ChestXRayDataset(train_dirs, test_transform)
    test_data_global = torch.utils.data.DataLoader(Subset(total_dataset, test_idx), batch_size=batch_size, shuffle=True, num_workers=10)
    logging.info("##########test sample number = %d" % (len(Subset(test_data_global, test_idx))))
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
    ) = load_partition_data_covidrad(
        0.1,
        20,
        32,
    )
