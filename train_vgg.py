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
from models import model as M
from optimizer import build_optimizer
from utils import seed_all, save_checkpoint, train_one_epoch, validate, get_grad_norm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import copy
from torchvision import models


parser = argparse.ArgumentParser(description="Train mix-VGG for AICAS2023")
parser.add_argument("--config", required=True)
parser.add_argument("--dataset", default="cifar10",required=False)
args = parser.parse_args()

config = EasyDict(yaml.full_load(open(args.config)))


def add_scalars(
    writer,
    epoch,
    loss_val,
    acc_val,
):
    writer.add_scalar('Loss/Val', loss_val, epoch)
    writer.add_scalar('Accuracy/Val', acc_val, epoch)


def repvgg_model_convert(model, do_copy=True):
    if do_copy:
        deploy_model = copy.deepcopy(model)
    for module in deploy_model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    return deploy_model


def main(config, dataset):
    os.makedirs(config.output.dir, exist_ok=True)
    os.makedirs(os.path.join(config.output.dir, 'runs'), exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join(config.output.dir,
                                                'runs/result'),
                           flush_secs=30)
    logger = create_logger(config.output.dir, config.output.name)
    cifar10_data_dir = '~/data/pytorch_cifar10'
    cifar100_data_dir = '~/data/pytorch_cifar100'
    # load dataset
    # if args.dataset == "cifar10":
    #     transforms_normalize = transforms.Normalize(
    #         mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    #     transform_list = [transforms.ToTensor(), transforms_normalize]
    #     transformer = transforms.Compose(transform_list)
    # else:
    #     pass
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

    if args.dataset == 'cifar10':
        num_classes = 10
        train_dataset = datasets.CIFAR10(root=cifar10_data_dir,
                                         train=True,
                                         download=True,
                                         transform=transform_train)
        val_dataset = datasets.CIFAR10(root=cifar10_data_dir,
                                       train=False,
                                       download=True,
                                       transform=transform_test)
    elif args.dataset == 'cifar100':
        num_classes = 100
        train_dataset = datasets.CIFAR100(root=cifar100_data_dir,
                                          train=False,
                                          download=True,
                                          transform=transform_train)
        val_dataset = datasets.CIFAR100(root=cifar100_data_dir,
                                        train=False,
                                        download=True,
                                        transform=transform_test)
    else:
        raise ValueError('Unknown dataset_name=' + args.dataset)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.workers,
        pin_memory=True,
        sampler=None)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=config.val.batch_size,
                                             shuffle=False,
                                             num_workers=config.val.workers,
                                             pin_memory=True,
                                             sampler=None)

    # model = M.Net(config, num_classes)
    model = models.VGG11_BN_Weights(num_classes=num_classes)
    
    optimizer = build_optimizer(config, model)
    model.cuda()
    lr_scheduler = build_lr_scheduler(config, optimizer, len(train_loader))
    if config.train.loss == 'crossentropy':
        criterion = nn.CrossEntropyLoss()

    max_accuracy = 0.0
    min_loss = float("inf")
    logger.info("Start training...")
    start_time = time.time()
    for epoch in range(config.train.start_epoch, config.train.epochs):
        # train_one_epoch(config, model, criterion, train_loader, optimizer, epoch, lr_scheduler, logger)
        train_one_epoch(config,
                        model,
                        criterion,
                        train_loader,
                        optimizer,
                        epoch,
                        lr_scheduler,
                        logger,
                        writer=writer)
        if epoch % config.output.epoch_print_freq == 0 or epoch >= (
                config.train.epochs - 10):
            acc1, acc5, loss = validate(config, val_loader, model, logger,
                                        num_classes, epoch, writer)
            logger.info(
                f"Valdation set Accuracy of the network at epoch {epoch}: Top1: {acc1:.3f}%, Top5: {acc5:.3f}%, Loss: {loss:.5f}"
            )
            max_accuracy = max(max_accuracy, acc1)
            min_loss = min(min_loss, loss)
            logger.info(f'Max accuracy: {max_accuracy:.2f}%')
            logger.info(f'Min validate loss: {min_loss:.4f}')
            is_best = max_accuracy == acc1 or min_loss == loss
            add_scalars(writer, epoch, loss, acc1)
            if epoch % config.output.save_freq == 0 or is_best:
                save_checkpoint(config,
                                epoch,
                                model,
                                max_accuracy,
                                loss,
                                optimizer,
                                lr_scheduler,
                                logger,
                                is_best=is_best)
    save_checkpoint(config,
                    epoch,
                    model,
                    max_accuracy,
                    loss,
                    optimizer,
                    lr_scheduler,
                    logger,
                    is_best=is_best)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    seed_all()
    main(config, args.dataset)
