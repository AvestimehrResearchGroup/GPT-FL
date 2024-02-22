import torch
import os
from torchvision.datasets import VisionDataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

class CIFAR100GeneratedDataset(VisionDataset):

    def __init__(self, 
                 root: str,
                 transforms=None, 
                 transform=None, 
                 target_transform=None) -> None:
        super().__init__(root, transforms, transform, target_transform)

        self.samples = []

        folders = os.listdir(root)
        for folder in folders:
            _temp_files = []
            for f in os.listdir(os.path.join(root, folder)):
                _temp_files.append(os.path.join(root, folder, f))
            self.samples += _temp_files
            
        self.classes = ['apple','aquarium_fish','baby','bear','beaver','bed','bee',
                        'beetle','bicycle','bottle','bowl','boy','bridge','bus',
                        'butterfly','camel','can','castle','caterpillar','cattle','chair',
                        'chimpanzee','clock','cloud','cockroach','couch','crab','crocodile',
                        'cup','dinosaur','dolphin','elephant','flatfish','forest','fox',
                        'girl','hamster','house','kangaroo','keyboard','lamp','lawn_mower',
                        'leopard','lion','lizard','lobster','man','maple_tree','motorcycle',
                        'mountain','mouse','mushroom','oak_tree','orange','orchid','otter',
                        'palm_tree','pear','pickup_truck','pine_tree','plain','plate','poppy',
                        'porcupine','possum','rabbit','raccoon','ray','road','rocket',
                        'rose','sea','seal','shark','shrew','skunk','skyscraper',
                        'snail','snake','spider','squirrel','streetcar','sunflower','sweet_pepper',
                        'table','tank','telephone','television','tiger','tractor','train',
                        'trout','tulip','turtle','wardrobe','whale','willow_tree','wolf','woman','worm']


    def __getitem__(self, index: int) -> torch.Tensor:
        image = np.array(Image.open(self.samples[index]))
        image = torch.Tensor(image).float()
        image = torch.einsum("hwc->chw", image)[0:3,...]
        # image = image.reshape((3, 256, 256))
        image /= 255


        label = self.classes.index(self.samples[index].split("/")[-2])
        
        if self.transforms is not None:
            image = self.transforms(image)

        return image, label

        
    def __len__(self) -> int:
        return len(self.samples)

def load_partition_sync_data_cifar100(
    image_paths, batch_size
):
    print("############loading sync cifar100 dataset ################")
    transform = transforms.Compose([
                                transforms.RandomResizedCrop(32),
                                transforms.RandomHorizontalFlip(),
                                # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    cifar100_set = CIFAR100GeneratedDataset(root=image_paths, transforms=transform)
    # sync_data_loader = torch.utils.data.DataLoader(cifar100_set, batch_size=batch_size, shuffle=True, num_workers=16)

    return cifar100_set