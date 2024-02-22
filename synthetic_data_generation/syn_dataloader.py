import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import glob
import timm
from timm.data import transforms_factory
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torch.utils.data import DataLoader
import numpy as np

class SynCifar10Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = os.listdir(data_dir)
        self.targets = self._extract_labels()

    def __getitem__(self, index):
        image_path = os.path.join(self.data_dir, self.image_files[index])
        # image = Image.open(image_path).convert('RGB')
        # image = np.asarray(np.uint8(image))

        image = np.array(Image.open(image_path))
        image = torch.Tensor(image).float()
        image = torch.einsum("hwc->chw", image)[0:3,...]
        image /= 255

        if "Airplane" in image_path or "airplane" in image_path:
            label = 0
        elif "Automobile" in image_path or "automobile" in image_path:
            label = 1
        elif "Bird" in image_path or "bird" in image_path:
            label = 2
        elif "Cat" in image_path or "cat" in image_path:
            label = 3
        elif "Deer" in image_path or "deer" in image_path:
            label = 4
        elif "Dog" in image_path or "dog" in image_path:
            label = 5
        elif "Frog" in image_path or "frog" in image_path:
            label = 6
        elif "Horse" in image_path or "horse" in image_path:
            label = 7
        elif "Ship" in image_path or "ship" in image_path:
            label = 8
        elif "Truck" in image_path or "truck" in image_path:
            label = 9
        # image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.image_files)

    # def targets(self):
    #     return self.labels  # Provide access to the labels through a property

    def _extract_labels(self):
        labels = []
        for image_file in self.image_files:
            if "Airplane" in image_file or "airplane" in image_file:
                labels.append(0)
            elif "Automobile" in image_file or "automobile" in image_file:
                labels.append(1)
            elif "Bird" in image_file or "bird" in image_file:
                labels.append(2)
            elif "Cat" in image_file or "cat" in image_file:
                labels.append(3)
            elif "Deer" in image_file or "deer" in image_file:
                labels.append(4)
            elif "Dog" in image_file or "dog" in image_file:
                labels.append(5)
            elif "Frog" in image_file or "frog" in image_file:
                labels.append(6)
            elif "Horse" in image_file or "horse" in image_file:
                labels.append(7)
            elif "Ship" in image_file or "ship" in image_file:
                labels.append(8)
            elif "Truck" in image_file or "truck" in image_file:
                labels.append(9)
            else:
                labels.append(-1)  # Use -1 or any other convention for unknown labels
        return labels

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

class SynFlower102Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = os.listdir(data_dir)

    def __getitem__(self, index):
        image_path = os.path.join(self.data_dir, self.image_files[index])
        image = Image.open(image_path).convert('RGB')
        # image = image.resize((32, 32))

        if "fire_lily" in image_path:
            label = 21
        elif "canterbury_bells" in image_path:
            label = 3
        elif "bolero_deep_blue" in image_path:
            label = 45
        elif "pink_primrose" in image_path:
            label = 1
        elif "mexican_aster" in image_path:
            label = 34
        elif "prince_of_wales_feathers" in image_path:
            label = 27
        elif "moon_orchid" in image_path:
            label = 7
        elif "globe-flower" in image_path:
            label = 16
        elif "grape_hyacinth" in image_path:
            label = 25
        elif "corn_poppy" in image_path:
            label = 26
        elif "toad_lily" in image_path:
            label = 79
        elif "siam_tulip" in image_path:
            label = 39
        elif "red_ginger" in image_path:
            label = 24
        elif "spring_crocus" in image_path:
            label = 67
        elif "alpine_sea_holly" in image_path:
            label = 35
        elif "garden_phlox" in image_path:
            label = 32
        elif "globe_thistle" in image_path:
            label = 10
        elif "tiger_lily" in image_path:
            label = 6
        elif "ball_moss" in image_path:
            label = 93
        elif "love_in_the_mist" in image_path:
            label = 33
        elif "monkshood" in image_path:
            label = 9
        elif "blackberry_lily" in image_path:
            label = 102
        elif "spear_thistle" in image_path:
            label = 14
        elif "balloon_flower" in image_path:
            label = 19
        elif "blanket_flower" in image_path:
            label = 100
        elif "king_protea" in image_path:
            label = 13
        elif "oxeye_daisy" in image_path:
            label = 49
        elif "yellow_iris" in image_path:
            label = 15
        elif "cautleya_spicata" in image_path:
            label = 61
        elif "carnation" in image_path:
            label = 31
        elif "silverbush" in image_path:
            label = 64
        elif "bearded_iris" in image_path:
            label = 68
        elif "black-eyed_susan" in image_path:
            label = 63
        elif "windflower" in image_path:
            label = 69
        elif "japanese_anemone" in image_path:
            label = 62
        elif "giant_white_arum_lily" in image_path:
            label = 20
        elif "great_masterwort" in image_path:
            label = 38
        elif "sweet_pea" in image_path:
            label = 4
        elif "tree_mallow" in image_path:
            label = 86
        elif "trumpet_creeper" in image_path:
            label = 101
        elif "daffodil" in image_path:
            label = 42
        elif "pincushion_flower" in image_path:
            label = 22
        elif "hard-leaved_pocket_orchid" in image_path:
            label = 2
        elif "sunflower" in image_path:
            label = 54
        elif "osteospermum" in image_path:
            label = 66
        elif "tree_poppy" in image_path:
            label = 70
        elif "desert-rose" in image_path:
            label = 85
        elif "bromelia" in image_path:
            label = 99
        elif "magnolia" in image_path:
            label = 87
        elif "english_marigold" in image_path:
            label = 5
        elif "bee_balm" in image_path:
            label = 92
        elif "stemless_gentian" in image_path:
            label = 28
        elif "mallow" in image_path:
            label = 97
        elif "gaura" in image_path:
            label = 57
        elif "lenten_rose" in image_path:
            label = 40
        elif "marigold" in image_path:
            label = 47
        elif "orange_dahlia" in image_path:
            label = 59
        elif "buttercup" in image_path:
            label = 48
        elif "pelargonium" in image_path:
            label = 55
        elif "ruby-lipped_cattleya" in image_path:
            label = 36
        elif "hippeastrum" in image_path:
            label = 91
        elif "artichoke" in image_path:
            label = 29
        elif "gazania" in image_path:
            label = 71
        elif "canna_lily" in image_path:
            label = 90
        elif "peruvian_lily" in image_path:
            label = 18
        elif "mexican_petunia" in image_path:
            label = 98
        elif "bird_of_paradise" in image_path:
            label = 8
        elif "sweet_william" in image_path:
            label = 30
        elif "purple_coneflower" in image_path:
            label = 17
        elif "wild_pansy" in image_path:
            label = 52
        elif "columbine" in image_path:
            label = 84
        elif "colt's_foot" in image_path:
            label = 12
        elif "snapdragon" in image_path:
            label = 11
        elif "camellia" in image_path:
            label = 96
        elif "fritillary" in image_path:
            label = 23
        elif "common_dandelion" in image_path:
            label = 50
        elif "poinsettia" in image_path:
            label = 44
        elif "primula" in image_path:
            label = 53
        elif "azalea" in image_path:
            label = 72
        elif "californian_poppy" in image_path:
            label = 65
        elif "anthurium" in image_path:
            label = 80
        elif "morning_glory" in image_path:
            label = 76
        elif "cape_flower" in image_path:
            label = 37
        elif "bishop_of_llandaff" in image_path:
            label = 56
        elif "pink-yellow_dahlia" in image_path:
            label = 60
        elif "clematis" in image_path:
            label = 82
        elif "geranium" in image_path:
            label = 58
        elif "thorn_apple" in image_path:
            label = 75
        elif "barbeton_daisy" in image_path:
            label = 41
        elif "bougainvillea" in image_path:
            label = 95
        elif "sword_lily" in image_path:
            label = 43
        elif "hibiscus" in image_path:
            label = 83
        elif "lotus" in image_path:
            label = 78
        elif "cyclamen" in image_path:
            label = 88
        elif "foxglove" in image_path:
            label = 94
        elif "frangipani" in image_path:
            label = 81
        elif "rose" in image_path:
            label = 74
        elif "watercress" in image_path:
            label = 89
        elif "water_lily" in image_path:
            label = 73
        elif "wallflower" in image_path:
            label = 46
        elif "passion_flower" in image_path:
            label = 77
        elif "petunia" in image_path:
            label = 51

        if self.transform is not None:
            image = self.transform(image)
        # print(label)
        return image, label-1

    def __len__(self):
        return len(self.image_files)
    
class SynFood101Dataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        data_dir, 
        transform=None, 
        class_to_idx=None,
        model=None,
    ):
        self.data_dir = data_dir
        self.transform = transform
        self.class_to_idx = class_to_idx
        # self.image_files = os.listdir(data_dir)
        self.image_files = glob.glob(f'{data_dir}*/*.png')
        self.model = model
        
        """
        for index in range(len(self.image_files)):
            image_file_path = self.image_files[index].split("train_dataset/")[1]
            image_path = os.path.join(self.data_dir, image_file_path)
        
            try:
                print(f"{image_path}")
                image = Image.open(image_path).convert("RGB")
            except Exception as err:
                print(f"{image_path}")
                raise
        """

    def __getitem__(self, index):
        # Read image data, mobilevit use BGR input
        image_file_path = self.image_files[index].split("train_dataset/")[1]
        image_path = os.path.join(self.data_dir, image_file_path)
        # print(image_path)
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as err:
            print(f"{image_path}")
            raise
        
        label_str = image_file_path.split("_train")[0].split("/")[1]
        label = self.class_to_idx[label_str]
        
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.image_files)

class SynCovidDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = os.listdir(data_dir)

    def __getitem__(self, index):
        image_path = os.path.join(self.data_dir, self.image_files[index])
        image = Image.open(image_path).convert('RGB')
        # image = image.resize((32, 32))

        if "NORMAL" in image_path:
            label = 0
        elif "Viral Pneumonia" in image_path:
            label = 1
        elif "COVID" in image_path:
            label = 2

        if self.transform is not None:
            image = self.transform(image)
        # print(label)
        return image, label

    def __len__(self):
        return len(self.image_files)



# image_paths = "syn_dataset/flower102/syn_dataset"

# model = timm.create_model('resnet18', pretrained=False)
# config = resolve_data_config({}, model=model)
# transform = create_transform(**config)

# dataset = SynFlower102Dataset(image_paths, transform=transform)

# for i in range(len(dataset)):
#     image, label = dataset[i]
