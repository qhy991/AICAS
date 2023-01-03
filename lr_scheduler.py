from timm.scheduler.cosine_lr import CosineLRScheduler


def build_lr_scheduler(config, optimizer, n_iter_per_epoch):
    num_steps = int(config.train.epochs * n_iter_per_epoch)
    warmup_steps = int(config.train.warmup_epochs * n_iter_per_epoch)
    # decay_steps = int(config.train.lr_scheduler.decay_epochs * n_iter_per_epoch)

    lr_scheduler = CosineLRScheduler(
        optimizer=optimizer,
        t_initial=num_steps,
        lr_min=config.train.lr_scheduler.min_lr,
        warmup_lr_init=config.train.warmup_lr,
        warmup_t=warmup_steps,
        cycle_limit=1,
        t_in_epochs=False,
    )

    return lr_scheduler