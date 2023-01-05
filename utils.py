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
            logger.info(
                f'Train: [{epoch}/{config.train.epochs}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
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
        lr_scheduler.step_update(epoch * num_steps + idx)

        batch_time.update(time.time() - end)
        loss_meter.update(loss.item(), labels.size(0))
        norm_meter.update(grad_norm)

        end = time.time()

        if idx % config.output.print_freq == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.train.epochs}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
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
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
               'ship', 'truck')
    N_CLASSES = 10
    class_correct = list(0. for i in range(N_CLASSES))
    class_total_1 = list(0. for i in range(N_CLASSES))

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
            logger.info(f'Test: [{idx}/{len(val_loader)}]\t'
                        f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                        f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                        f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                        f'Mem {memory_used:.0f}MB')
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    logger.info(f' * per class {np.array(class_correct)/np.array(class_total_1)} ')
    logger.info(class_total)
    for i in range(N_CLASSES):
        writer.add_scalar('class-Acc/' + classes[i],
                          class_correct[i] / class_total_1[i], epoch)

    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg
