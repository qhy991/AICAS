import argparse
import yaml
from easydict import EasyDict
import time
import datetime
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter
import numpy as np
from data import build_loader, build_hs_loader
from logger import create_logger
from lr_scheduler import build_lr_scheduler
from models import model_mask as M
from optimizer import build_optimizer
from utils import seed_all, save_checkpoint, train_one_epoch, validate, get_grad_norm,repvgg_model_convert,get_dataset
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import copy
import tqdm
parser = argparse.ArgumentParser(description="Train mix-VGG for AICAS2023")
parser.add_argument("--config", required=True)
parser.add_argument("--dataset", default="mask",required=False)
parser.add_argument("--finetune", default=False,required=False)

args = parser.parse_args()

config = EasyDict(yaml.full_load(open(args.config)))

NO_num = int(args.config.split('/')[-1].split('-')[0])
print(NO_num)
def add_scalars(
    writer,
    epoch,
    loss_val,
    acc_val,
):
    if writer != None:
        writer.add_scalar('Loss/Val', loss_val, epoch)
        writer.add_scalar('Accuracy/Val', acc_val, epoch)



def main(config, dataset,NO):
    os.makedirs(config.output.dir, exist_ok=True)
    os.makedirs(os.path.join(config.output.dir, 'runs'), exist_ok=True)

    # writer = SummaryWriter(log_dir=os.path.join(config.output.dir,
    #                                             'runs/result'),
    #                        flush_secs=30)
    writer = None
    logger = create_logger(config.output.dir, config.output.name)
    # train_loader,val_loader,num_classes = get_dataset(config)
    mean = [0.5364829, 0.47852907, 0.45479727]
    std = [0.28900605, 0.28060046, 0.28859267]
    resolution = config.model.resolution
    transform_train = transforms.Compose([
        transforms.Resize((resolution,resolution)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    num_classes = 2
    print("resolution:",resolution)
    train_dataset = datasets.ImageFolder("/home/qhy/data/Face Mask Dataset/Train",transform=transform_train)
    # val_dataset = datasets.ImageFolder("/home/qhy/data/Face Mask Dataset/Test",transform=transform_train) 
    # val_dataset = datasets.ImageFolder("/home/qhy/data/mask_classification/train",transform=transform_train) 
    val_dataset_1 = datasets.ImageFolder("/home/qhy/data/mask_classification/train",transform=transform_train) 
    val_dataset_2 = datasets.ImageFolder("/home/qhy/data/Face Mask Dataset/Test",transform=transform_train) 
    val_dataset_3 = datasets.ImageFolder("/home/qhy/data/mask_classification/test",transform=transform_train) 
    

    def merge_datasets(dataset, sub_dataset):
        '''
            需要合并的Attributes:
                classes (list): List of the class names sorted alphabetically.
                class_to_idx (dict): Dict with items (class_name, class_index).
                samples (list): List of (sample path, class_index) tuples
                targets (list): The class_index value for each image in the dataset
        '''
        # 合并 classes
        dataset.classes.extend(sub_dataset.classes)
        dataset.classes = sorted(list(set(dataset.classes)))
        # 合并 class_to_idx
        dataset.class_to_idx.update(sub_dataset.class_to_idx)
        # 合并 samples
        dataset.samples.extend(sub_dataset.samples)
        # 合并 targets
        dataset.targets.extend(sub_dataset.targets)
        return dataset
    val_dataset_temp = merge_datasets(val_dataset_1,val_dataset_2)
    val_dataset = merge_datasets(val_dataset_temp,val_dataset_3)


    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        sampler=None)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        sampler=None)


    model = M.Net(config, num_classes)
    
    # optimizer =  build_optimizer(config, model)
    optimizer = torch.optim.SGD(model.parameters(),lr=0.02)
    
    model.cuda()
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=0.001)

    loss_func = torch.nn.CrossEntropyLoss()

    n_epochs = 60
    predict_acc = []
    train_acc = []
    best_acc = 0
    for epoch in range(n_epochs):
        running_loss = 0.0
        running_correct = 0.0
        print("Epoch  {}/{}".format(epoch, n_epochs))
        num_iter = 0
        for data in tqdm.tqdm(train_loader):
            num_iter += 1
            X_train, y_train = data
            X_train, y_train = X_train.cuda(), y_train.cuda()
            outputs = model(X_train)
            _, pred = torch.max(outputs.data, 1)
            optimizer.zero_grad()
            loss = loss_func(outputs, y_train)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            running_correct += torch.sum(pred == y_train.data)
            if num_iter > 200:
                break
        testing_correct = 0.0
        for data in tqdm.tqdm(val_loader):
            X_test, y_test = data
            # X_test, y_test = Variable(X_test), Variable(y_test)
            X_test, y_test = X_test.cuda(), y_test.cuda()
            outputs = model(X_test)
            _, pred = torch.max(outputs, 1) #返回每一行中最大值的那个元素，且返回其索引
            testing_correct += torch.sum(pred == y_test.data)
            # print(testing_correct)
        test_Acc = 100 * testing_correct / len(val_dataset)
        train_Acc = 100 * running_correct / len(train_dataset)
        print("lr is :{:.8f} Loss is :{:.4f},Train Accuracy is:{:.4f}%,Test Accuracy is:{:.4f}%".format(
            lr_scheduler.get_last_lr()[0],
            running_loss / len(train_dataset), train_Acc,
            test_Acc))
        if(test_Acc>best_acc):
            best_acc=test_Acc
            best_model = model
        lr_scheduler.step()
        predict_acc.append(100 * testing_correct / len(val_dataset))
        train_acc.append(100 * running_correct / len(train_dataset))
    max_accuracy = max(predict_acc)
    save_checkpoint(config,
                    epoch,
                    best_model,
                    max_accuracy,
                    loss,
                    optimizer,
                    lr_scheduler,
                    logger,
                    is_best=True)
    save_content = "|{}|{}|{}|".format(NO,resolution,max_accuracy)
    f = open("/home/qhy/Reserach/AICAS/mask_exp_note.md","a")
    print(save_content,file=f)
    f.close()


if __name__ == "__main__":
    seed_all()
    main(config, args.dataset,NO_num)
