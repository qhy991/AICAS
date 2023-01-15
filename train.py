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
from utils import seed_all, save_checkpoint, train_one_epoch, validate, get_grad_norm,repvgg_model_convert,get_dataset
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import copy

parser = argparse.ArgumentParser(description="Train mix-VGG for AICAS2023")
parser.add_argument("--config", required=True)
parser.add_argument("--dataset", default="cifar10",required=False)
parser.add_argument("--finetune", default=False,required=False)

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



def main(config, dataset):
    os.makedirs(config.output.dir, exist_ok=True)
    os.makedirs(os.path.join(config.output.dir, 'runs'), exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join(config.output.dir,
                                                'runs/result'),
                           flush_secs=30)
    logger = create_logger(config.output.dir, config.output.name)
    train_loader,val_loader,num_classes = get_dataset(config)

    model = M.Net(config, num_classes)
    if args.finetune:
        if config.dataset == "cifar10":
            # model_path = "/home/qhy/Reserach/AICAS/log/cifar10/" + args.config.split("/")[-1].replace(".yaml","/best_ckpt.pth")
            model_path = "/home/qhy/Reserach/AICAS/log/search-best/" + args.config.split("/")[-1].replace(".yaml","/best_ckpt.pth")
            
        elif config.dataset == "cifar100":
            model_path = "/home/qhy/Reserach/AICAS/log/cifar100/" + args.config.split("/")[-1].replace(".yaml","/best_ckpt.pth")
        model.load_state_dict(torch.load(model_path)['model'])
        print("load successfully!")
    optimizer = build_optimizer(config, model)
    model.cuda()
    if args.finetune:
        lr_scheduler = build_lr_scheduler(config, optimizer, len(train_loader),"cos")
    else:
        lr_scheduler = build_lr_scheduler(config, optimizer, len(train_loader),"cos")
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
