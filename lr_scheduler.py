from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler

def build_lr_scheduler(config, optimizer, n_iter_per_epoch,scheduler="cos"):
    num_steps = int(config.train.epochs * n_iter_per_epoch)
    warmup_steps = int(config.train.warmup_epochs * n_iter_per_epoch)
    # decay_steps = int(config.train.lr_scheduler.decay_epochs * n_iter_per_epoch)
    if scheduler=="cos":
        lr_scheduler = CosineLRScheduler(
            optimizer=optimizer,
            t_initial=num_steps,
            lr_min=config.train.lr_scheduler.min_lr,
            warmup_lr_init=config.train.warmup_lr,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
        )
    elif scheduler=="step":
        lr_scheduler = StepLRScheduler(
            optimizer=optimizer,
            decay_t = 10, 
            decay_rate=0.5
        )

    return lr_scheduler