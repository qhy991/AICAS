import torch
import os
import random
import numpy as np
import time
import logging
import datetime
from timm.utils import accuracy, AverageMeter
import torch.nn as nn
from models.RepVGGBlock import RepVGGBlock
from data import mixup_data, mixup_criterion
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import copy
def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# class AverageMeter(object):
#     """Computes and stores the average and current value"""

#     def __init__(self, name, fmt=':f'):
#         self.name = name
#         self.fmt = fmt
#         self.reset()

#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0

#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count

#     def __str__(self):
#         fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
#         return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


# def accuracy(output, target, topk=(1,)):
#     """Computes the accuracy over the k top predictions for the specified values of k"""
#     with torch.no_grad():
#         maxk = max(topk)
#         batch_size = target.size(0)

#         _, pred = output.topk(maxk, 1, True, True)
#         pred = pred.t()
#         correct = pred.eq(target.view(1, -1).expand_as(pred))

#         res = []
#         for k in topk:
#             correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
#             res.append(correct_k.mul_(100.0 / batch_size))
#         return res

# @torch.no_grad()
# def validate_model(val_loader, model, device=None, print_freq=100):
#     if device is None:
#         device = next(model.parameters()).device
#     else:
#         model.to(device)
#     batch_time = AverageMeter('Time', ':6.3f')
#     top1 = AverageMeter('Acc@1', ':6.2f')
#     top5 = AverageMeter('Acc@5', ':6.2f')
#     progress = ProgressMeter(
#         len(val_loader),
#         [batch_time, top1, top5],
#         prefix='Test: ')

#     # switch to evaluate mode
#     model.eval()

#     end = time.time()
#     for i, (images, target) in enumerate(val_loader):
#         images = images.to(device)
#         target = target.to(device)

#         # compute output
#         output = model(images)

#         # measure accuracy and record loss
#         acc1, acc5 = accuracy(output, target, topk=(1, 5))
#         top1.update(acc1[0], images.size(0))
#         top5.update(acc5[0], images.size(0))

#         # measure elapsed time
#         batch_time.update(time.time() - end)
#         end = time.time()

#         if i % print_freq == 0:
#             progress.display(i)

#     # print(
#     #     ' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
#     logging.info(
#         ' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
#     return top1.avg, top5.avg


def get_train_samples(train_loader, num_samples):
    train_data = []
    for batch in train_loader:
        train_data.append(batch[0])
        if len(train_data) * batch[0].size(0) >= num_samples:
            break
    return torch.cat(train_data, dim=0)[:num_samples]


def save_checkpoint(config,
                    epoch,
                    model,
                    max_accuracy,
                    loss,
                    optimizer,
                    lr_scheduler,
                    logger,
                    is_best=False):
    save_state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'max_accuracy': max_accuracy,
        'loss': loss,
        'epoch': epoch,
        'config': config
    }

    if is_best:
        best_path = os.path.join(config.output.dir, 'best_ckpt.pth')
        logger.info(f"{best_path} saving best model......")
        torch.save(save_state, best_path)
        logger.info(f"{best_path} best model saved.")

    save_path = os.path.join(config.output.dir, f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item()**norm_type
    total_norm = total_norm**(1. / norm_type)
    return total_norm


def finetune_one_epoch(config, model, criterion, train_loader, optimizer,
                       epoch, lr_scheduler, logger, add_RSG, writer):
    criterion.cuda()
    model.train()
    optimizer.zero_grad()
    num_steps = len(train_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    start = time.time()
    end = time.time()
    for idx, (imgs, labels) in enumerate(train_loader):
        # print(labels)
        imgs, labels = imgs.cuda(), labels.cuda()
        # outputs, cesc_loss, total_mv_loss, combine_target= model(imgs,labels, epoch, add_RSG)
        # loss = criterion(outputs, combine_target) + 0.1 * cesc_loss.mean() + 0.01 * total_mv_loss.mean()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        # print(loss)
        if config.train.use_l2_norm:
            for module in model.modules():
                if isinstance(module, RepVGGBlock):
                    loss = loss + config.train.optimizer.weight_decay_param.decay * 0.5 * module.get_custom_L2(
                    )

        optimizer.zero_grad()
        loss.backward()
        if writer != None:
            writer.add_scalar('Loss/Train', loss, epoch)
            writer.add_scalar('Lr/base',
                            optimizer.state_dict()['param_groups'][0]['lr'],
                            epoch)
            writer.add_scalar('Lr/repvgg',
                            optimizer.state_dict()['param_groups'][1]['lr'],
                            epoch)

        if 'hs' in config.model:
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(),
                                                 config.train.hs.clip_grad)
        else:
            grad_norm = get_grad_norm(model.parameters())

        optimizer.step()
        # lr_scheduler.step_update(epoch * num_steps + idx)

        batch_time.update(time.time() - end)
        loss_meter.update(loss.item(), labels.size(0))
        norm_meter.update(grad_norm)

        end = time.time()

        if idx % config.output.print_freq == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            if logger!= None:
                logger.info(
                    f'Train: [{epoch}/{config.train.epochs}][{idx}/{num_steps}]\t'
                    f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                    f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                    f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                    f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                    f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    if logger!= None:
        logger.info(
            f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}"
        )


def train_one_epoch(config, model, criterion, train_loader, optimizer, epoch,
                    lr_scheduler, logger, writer):
    criterion.cuda()
    model.train()
    optimizer.zero_grad()
    num_steps = len(train_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    start = time.time()
    end = time.time()
    for idx, (imgs, labels) in enumerate(train_loader):
        # print(labels)
        imgs, labels = imgs.cuda(), labels.cuda()

        outputs = model(imgs)
        # print(outputs.shape)
        # print(labels.shape)
        loss = criterion(outputs.squeeze(-1).squeeze(-1), labels)

        # if config.data.mixup.alpha > 0 and epoch < config.train.mix_up_epoch:
        #     mixed_images, labels_a, labels_b, lam = mixup_data(imgs, labels, config.data.mixup.alpha)
        #     outputs, cesc_loss, total_mv_loss, combine_target= model(mixed_images,labels, epoch, add_RSG)
        #     loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        # else:
        #     outputs, cesc_loss, total_mv_loss, combine_target= model(imgs,labels, epoch, add_RSG)
        #     loss = criterion(outputs, combine_target) + 0.1 * cesc_loss.mean() + 0.01 * total_mv_loss.mean()
        # outputs, cesc_loss, total_mv_loss, combine_target= model(imgs,labels, epoch, add_RSG)
        # loss = criterion(outputs, combine_target) + 0.1 * cesc_loss.mean() + 0.01 * total_mv_loss.mean()
        # outputs = model(imgs)
        # loss = criterion(outputs, labels)

        if config.train.use_l2_norm:
            for module in model.modules():
                if isinstance(module, RepVGGBlock):
                    loss = loss + config.train.optimizer.weight_decay_param.repvgg_decay * 0.5 * module.get_custom_L2(
                    )

        optimizer.zero_grad()
        loss.backward()
        if writer!=None:
            writer.add_scalar('Loss/Train', loss, epoch)
            writer.add_scalar('Lr/base',
                            float(optimizer.state_dict()['param_groups'][0]['lr']),
                            epoch)
            writer.add_scalar('Lr/repvgg',
                            float(optimizer.state_dict()['param_groups'][1]['lr']),
                            epoch)

        if 'hs' in config.model:
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(),
                                                 config.train.hs.clip_grad)
        else:
            grad_norm = get_grad_norm(model.parameters())

        optimizer.step()
        lr_scheduler.step_update(epoch * num_steps + idx)

        batch_time.update(time.time() - end)
        loss_meter.update(loss.item(), labels.size(0))
        norm_meter.update(grad_norm)

        end = time.time()

        if idx % config.output.print_freq == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            if logger!= None:
                logger.info(
                    f'Train: [{epoch}/{config.train.epochs}][{idx}/{num_steps}]\t'
                    f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                    f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                    f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                    f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                    f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    if logger!= None:
        logger.info(
            f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}"
        )


def accuracy_per_class(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [
        correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size
        for k in topk
    ]


@torch.no_grad()
def validate(config, val_loader, model, logger, num_classes, epoch, writer):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    if num_classes ==10 :
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
               'ship', 'truck')
    else :
        classes = {19: 'cattle', 29: 'dinosaur', 0: 'apple', 11: 'boy', 1: 'aquarium_fish', 86: 'telephone', 90: 'train', 28: 'cup', 23: 'cloud', 31: 'elephant', 39: 'keyboard', 96: 'willow_tree', 82: 'sunflower', 17: 'castle', 71: 'sea', 8: 'bicycle', 97: 'wolf', 80: 'squirrel', 74: 'shrew', 59: 'pine_tree', 70: 'rose', 87: 'television', 
         84: 'table', 64: 'possum', 52: 'oak_tree', 42: 'leopard', 47: 'maple_tree', 65: 'rabbit', 21: 'chimpanzee', 22: 'clock', 81: 'streetcar', 24: 'cockroach', 78: 'snake', 45: 'lobster', 49: 'mountain', 56: 'palm_tree', 76: 'skyscraper', 89: 'tractor', 73: 'shark', 14: 'butterfly', 9: 'bottle', 6: 'bee', 20: 'chair', 98: 'woman', 
         36: 'hamster', 55: 'otter', 72: 'seal', 43: 'lion', 51: 'mushroom', 35: 'girl', 83: 'sweet_pepper', 33: 'forest', 27: 'crocodile', 53: 'orange', 92: 'tulip', 50: 'mouse', 15: 'camel', 18: 'caterpillar', 46: 'man', 75: 'skunk', 38: 'kangaroo', 66: 'raccoon', 77: 'snail', 69: 'rocket', 95: 'whale', 99: 'worm', 93: 'turtle',
         4: 'beaver', 61: 'plate', 94: 'wardrobe', 68: 'road', 34: 'fox', 32: 'flatfish', 88: 'tiger', 67: 'ray', 30: 'dolphin', 62: 'poppy', 63: 'porcupine', 40: 'lamp', 26: 'crab', 48: 'motorcycle', 79: 'spider', 85: 'tank', 54: 'orchid', 44: 'lizard', 7: 'beetle', 12: 'bridge', 2: 'baby', 41: 'lawn_mower', 37: 'house', 13: 'bus', 
         25: 'couch', 10: 'bowl', 57: 'pear', 5: 'bed', 60: 'plain', 91: 'trout', 3: 'bear', 58: 'pickup_truck', 16: 'can'}

    # N_CLASSES = 10
    class_correct = list(0. for i in range(num_classes))
    class_total_1 = list(0. for i in range(num_classes))

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()

    class_total = np.zeros(num_classes)
    class_true = np.zeros(num_classes)

    for idx, (imgs, labels) in enumerate(val_loader):
        imgs = imgs.cuda()
        labels = labels.cuda()
        outputs = model(imgs).squeeze(-1).squeeze(-1)
        loss = criterion(outputs, labels)
        acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
        for i in range(labels.size(0)):
            class_total[int(labels[i])] += 1
            if (int(labels[i]) == int(outputs.topk(1).indices[0])):
                class_true[int(labels[i])] += 1
        loss_meter.update(loss.item(), labels.size(0))
        acc1_meter.update(acc1.item(), labels.size(0))
        acc5_meter.update(acc5.item(), labels.size(0))
        pred = outputs.argmax(dim=1)
        c = (pred == labels).squeeze()

        for i in range(len(labels)):
            _label = labels[i]
            class_correct[_label] += c[i].item()
            class_total_1[_label] += 1

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.output.print_freq == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            if logger!= None:
                logger.info(f'Test: [{idx}/{len(val_loader)}]\t'
                            f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                            f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                            f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                            f'Mem {memory_used:.0f}MB')
    if logger!= None:
        logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
        logger.info(f' * per class {np.array(class_correct)/np.array(class_total_1)} ')
        logger.info(class_total)
    if num_classes == 10:
        for i in range(num_classes):
            if writer != None:
                writer.add_scalar('class-Acc/' + classes[i],
                                class_correct[i] / class_total_1[i], epoch)

    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg

def get_dataset(config):
    cifar10_data_dir = '~/data/pytorch_cifar10'
    cifar100_data_dir = '~/data/pytorch_cifar100'

    if config.dataset== "cifar10":
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
        
    else :
        mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
        std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
        transform_train = transforms.Compose([
            transforms.RandomCrop(36, padding=4),
            transforms.CenterCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        # transform_train = transforms.Compose([
        #     #transforms.ToPILImage(),
        #     transforms.RandomCrop(32, padding=4),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandomRotation(15),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean, std)
        # ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        # transform_test = transforms.Compose([
        #     transforms.RandomCrop(36, padding=4),
        #     transforms.CenterCrop(32),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean, std)
        # ])

    if config.dataset == 'cifar10':
        num_classes = 10
        train_dataset = datasets.CIFAR10(root=cifar10_data_dir,
                                            train=True,
                                            download=True,
                                            transform=transform_train)
        val_dataset = datasets.CIFAR10(root=cifar10_data_dir,
                                        train=False,
                                        download=True,
                                        transform=transform_test)
    elif config.dataset == 'cifar100':
        num_classes = 100
        train_dataset = datasets.CIFAR100(root=cifar100_data_dir,
                                            train=True,
                                            download=True,
                                            transform=transform_train)
        val_dataset = datasets.CIFAR100(root=cifar100_data_dir,
                                        train=False,
                                        download=True,
                                        transform=transform_test)

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
    return train_loader,val_loader,num_classes


def repvgg_model_convert(model, do_copy=True):
    if do_copy:
        deploy_model = copy.deepcopy(model)
    for module in deploy_model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    return deploy_model