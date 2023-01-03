import torch
import math
from torch.optim import Optimizer
import torch.optim as optim


class LAMB(Optimizer):
    """Implements a pure pytorch variant of FuseLAMB (NvLamb variant) optimizer from apex.optimizers.FusedLAMB
    reference: https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/Transformer-XL/pytorch/lamb.py # noqa
    LAMB was proposed in `Large Batch Optimization for Deep Learning: Training BERT in 76 minutes`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its norm. (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        grad_averaging (bool, optional): whether apply (1-beta2) to grad when
            calculating running averages of gradient. (default: True)
        max_grad_norm (float, optional): value used to clip global grad norm (default: 1.0)
        trust_clip (bool): enable LAMBC trust ratio clipping (default: False)
        always_adapt (boolean, optional): Apply adaptive learning rate to 0.0
            weight decay parameter (default: False)
    Example:
        >>> optimizer = LAMB(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-6)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """
    def __init__(self,
                 params,
                 lr=1e-3,
                 bias_correction=True,
                 betas=(0.9, 0.999),
                 eps=1e-6,
                 weight_decay=0.01,
                 grad_averaging=True,
                 max_grad_norm=1.0,
                 trust_clip=False,
                 always_adapt=False):
        defaults = dict(lr=lr,
                        bias_correction=bias_correction,
                        betas=betas,
                        eps=eps,
                        weight_decay=weight_decay,
                        grad_averaging=grad_averaging,
                        max_grad_norm=max_grad_norm,
                        trust_clip=trust_clip,
                        always_adapt=always_adapt)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        device = self.param_groups[0]['params'][0].device
        one_tensor = torch.tensor(
            1.0, device=device
        )  # because torch.where doesn't handle scalars correctly
        global_grad_norm = torch.zeros(1, device=device)
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        'Lamb does not support sparse gradients, consider SparseAdam instad.'
                    )
                global_grad_norm.add_(grad.pow(2).sum())

        global_grad_norm = torch.sqrt(global_grad_norm)
        # FIXME it'd be nice to remove explicit tensor conversion of scalars when torch.where promotes
        # scalar types properly https://github.com/pytorch/pytorch/issues/9190
        max_grad_norm = torch.tensor(self.defaults['max_grad_norm'],
                                     device=device)
        clip_global_grad_norm = torch.where(global_grad_norm > max_grad_norm,
                                            global_grad_norm / max_grad_norm,
                                            one_tensor)

        for group in self.param_groups:
            bias_correction = 1 if group['bias_correction'] else 0
            beta1, beta2 = group['betas']
            grad_averaging = 1 if group['grad_averaging'] else 0
            beta3 = 1 - beta1 if grad_averaging else 1.0

            # assume same step across group now to simplify things
            # per parameter step can be easily support by making it tensor, or pass list into kernel
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            if bias_correction:
                bias_correction1 = 1 - beta1**group['step']
                bias_correction2 = 1 - beta2**group['step']
            else:
                bias_correction1, bias_correction2 = 1.0, 1.0

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.div_(clip_global_grad_norm)
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient valuesa
                    state['exp_avg'] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=beta3)  # m_t
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad,
                                                value=1 - beta2)  # v_t

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(
                    group['eps'])
                update = (exp_avg / bias_correction1).div_(denom)

                weight_decay = group['weight_decay']
                if weight_decay != 0:
                    update.add_(p, alpha=weight_decay)

                if weight_decay != 0 or group['always_adapt']:
                    # Layer-wise LR adaptation. By default, skip adaptation on parameters that are
                    # excluded from weight decay, unless always_adapt == True, then always enabled.
                    w_norm = p.norm(2.0)
                    g_norm = update.norm(2.0)
                    # FIXME nested where required since logical and/or not working in PT XLA
                    trust_ratio = torch.where(
                        w_norm > 0,
                        torch.where(g_norm > 0, w_norm / g_norm, one_tensor),
                        one_tensor,
                    )
                    if group['trust_clip']:
                        # LAMBC trust clipping, upper bound fixed at one
                        trust_ratio = torch.minimum(trust_ratio, one_tensor)
                    update.mul_(trust_ratio)

                p.add_(update, alpha=-group['lr'])
        return loss


def build_optimizer(config, model):
    """
    Build optimizer, set weight decay of normalization to 0 by default.
    """
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        if config.train.use_l2_norm:
            skip_keywords = model.no_weight_decay_keywords(
                config.train.no_weight_decay)
        else:
            skip_keywords = model.no_weight_decay_keywords()
    parameters = set_weight_decay(
        model,
        skip,
        skip_keywords,
        echo=config.train.optimizer.weight_decay_param.echo)
    repvgg_stage = torch.nn.ModuleList()
    for stage in (model.stage_0, model.stage_1, model.stage_2, model.stage_3):
        for name, module in stage._modules.items():
            if "repvgg" in name:
                repvgg_stage.append(stage)
    repvgg_stage_params = list(map(id, repvgg_stage.parameters()))
    base_stage_param = filter(lambda p: id(p) not in repvgg_stage_params,
                              model.parameters())

    opt_lower = config.train.optimizer.name.lower()
    optimizer = None
    if opt_lower == 'sgd':
        optimizer = optim.SGD(
            [{
                'params':
                base_stage_param,
                'lr':
                config.train.optimizer.base_lr,
                "weight_decay":
                config.train.optimizer.weight_decay_param.base_decay
            }, {
                'params':
                repvgg_stage.parameters(),
                'lr':
                config.train.optimizer.repvgg_lr,
                "weight_decay":
                config.train.optimizer.weight_decay_param.repvgg_decay
            }],
            momentum=config.train.optimizer.momentum,
            nesterov=True,
        )
        # optimizer = optim.SGD(parameters, momentum=config.train.optimizer.momentum, nesterov=True,
        #                       lr=config.train.optimizer.base_lr, weight_decay=config.train.optimizer.weight_decay_param.decay)
        if config.train.optimizer.weight_decay_param.echo:
            print(
                '================================== SGD nest, momentum = {}, wd = {}'
                .format(config.train.optimizer.momentum,
                        config.train.optimizer.weight_decay_param.decay))
    elif opt_lower == 'adam':
        print('adam')
        optimizer = optim.Adam(
            parameters,
            eps=config.train.optimizer.eps,
            betas=config.train.optimizer.betas,
            lr=config.train.optimizer.base_lr,
            weight_decay=config.train.optimizer.weight_decay_param.decay)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(
            parameters,
            eps=config.train.optimizer.eps,
            betas=config.train.optimizer.betas,
            lr=config.train.optimizer.base_lr,
            weight_decay=config.train.optimizer.weight_decay_param.decay)
    elif opt_lower == 'lamb':
        optimizer = LAMB(
            parameters,
            lr=config.train.optimizer.base_lr,
            betas=config.train.optimizer.betas,
            weight_decay=config.train.optimizer.weight_decay_param.decay)

    return optimizer


def set_weight_decay(model, skip_list=(), skip_keywords=(), echo=False):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if 'identity.weight' in name:
            has_decay.append(param)
            if echo:
                print(f"{name} USE weight decay")
        elif len(param.shape) == 1 or name.endswith(".bias") or (
                name in skip_list) or check_keywords_in_name(
                    name, skip_keywords):
            no_decay.append(param)
            if echo:
                print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
            if echo:
                print(f"{name} USE weight decay")

    return [{'params': has_decay}, {'params': no_decay, 'weight_decay': 0.}]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin