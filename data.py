import torch
import torch as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, datasets
from PIL import Image
import os
import numpy as np


def get_data(root_dir, txt_name):
    fh = open(os.path.join(root_dir, txt_name))
    contexts = []
    for line in fh:
        line = line.rstrip()
        context = line.split()
        contexts.append((context[0], int(context[1])))
    fh.close()
    return contexts


class PracticalDataset(Dataset):
    """
        Dataset for AICAS
    """
    def __init__(self, dataset_dir, txt_file, transforms=None):
        super(PracticalDataset, self).__init__()
        self.dataset_dir = dataset_dir
        self.transforms = transforms
        self.txt_file = txt_file
        self.data = get_data(dataset_dir, txt_file)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.to_list()
        
        name, label = self.data[index]
        img_name = os.path.join(self.dataset_dir, name)
        img = Image.open(img_name).convert('RGB')

        if self.transforms is not None:
            img = self.transforms(img)
        
        return img, label


train_transforms = transforms.Compose([
    # transforms.Resize((64,64)),
    # transforms.CenterCrop((48,48)),
    transforms.RandomResizedCrop((48,48), scale=(0.08, 1)),
    # transforms.RandomResizedCrop((54,54), scale=(0.08, 1)),
    # transforms.RandomHorizontalFlip(),
    # transforms.ColorJitter(0.4, 0.4, 0.4),
    # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    transforms.ToTensor(),
    transforms.Normalize(
        # mean=[0.485, 0.456, 0.406], # imagenet pretrained statics
        # std=[0.229, 0.224, 0.225]
        mean=[0.4124, 0.3954, 0.4062], # imagenet pretrained statics
        std=[0.1708, 0.1775, 0.1798]
        
    )
])

val_transforms = transforms.Compose([
    # transforms.RandomResizedCrop((48,48), scale=(0.08, 1)),
    
    transforms.Resize((56,56)),
    # # transforms.Resize((60,60)),
    transforms.CenterCrop((48,48)),
    # transforms.CenterCrop((54,54)),
    transforms.ToTensor(),
    transforms.Normalize(
        # mean=[0.485, 0.456, 0.406], # imagenet pretrained statics
        # std=[0.229, 0.224, 0.225]image.png
        mean=[0.4124, 0.3954, 0.4062], # imagenet pretrained statics
        std=[0.1708, 0.1775, 0.1798]
    )
])


def build_loader(config):
    root_dir = config.data.dataset_path
    train_txt_file = config.data.train_file_name
    val_txt_file = config.data.val_file_name
    train_dataset = PracticalDataset(dataset_dir=root_dir, txt_file=train_txt_file, transforms=train_transforms)
    val_dataset = PracticalDataset(dataset_dir=root_dir, txt_file=val_txt_file, transforms=val_transforms)
    

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last = True,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.val.batch_size,
        num_workers=32,
        pin_memory=True,
    )
    return train_loader, val_loader


def build_hs_loader_cifar100(config):
    root_dir = config.data.hs.dataset_path

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
    ])

    trainset = datasets.CIFAR100(
        root=root_dir, train=True, download=False, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=config.train.hs.batch_size, shuffle=True, num_workers=4)

    testset = datasets.CIFAR100(
        root=root_dir, train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=config.val.hs.batch_size, shuffle=False, num_workers=4)
    return trainloader, testloader


def build_hs_loader(config):
    if config.data.hs.dataset == 'cifar100':
        return build_hs_loader_cifar100(config)
    else:
        return build_loader(config)

def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

if __name__ == "__main__":
    train_dataset = PracticalDataset(dataset_dir="/home/bjy/data/aaai2023_challenge", txt_file="train_90p.txt", transforms=train_transforms)
    print(train_dataset[0])