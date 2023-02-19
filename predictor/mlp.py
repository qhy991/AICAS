import numpy as np
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random
from timm.utils import accuracy, AverageMeter
import tqdm


def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_all()
# NOTE: faster on CPU
device = torch.device("cpu")


def accuracy_mse(prediction, target, scale=100.0):
    prediction = prediction.detach() * scale
    target = (target) * scale
    return F.mse_loss(prediction, target)


class FeedforwardNet(nn.Module):
    def __init__(
        self,
        input_dims: int = 5,
        num_layers: int = 3,
        layer_width: list = [10, 10, 10],
        output_dims: int = 1,
        activation="relu",
    ):
        super(FeedforwardNet, self).__init__()
        assert (len(layer_width) == num_layers), "number of widths should be \
        equal to the number of layers"

        self.activation = eval("F." + activation)

        all_units = [input_dims] + layer_width
        self.layers = nn.ModuleList([
            nn.Linear(all_units[i], all_units[i + 1])
            for i in range(num_layers)
        ])

        self.out = nn.Linear(all_units[-1], 1)

        # make the init similar to the tf.keras version
        for l in self.layers:
            torch.nn.init.xavier_uniform_(l.weight)
            torch.nn.init.zeros_(l.bias)
        torch.nn.init.xavier_uniform_(self.out.weight)
        torch.nn.init.zeros_(self.out.bias)

    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        return self.out(x)

    def basis_funcs(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        return


def pair_loss(outputs, labels):
    output = outputs.unsqueeze(1)
    output1 = output.repeat(1, outputs.shape[0])
    label = labels.unsqueeze(1)
    label1 = label.repeat(1, labels.shape[0])
    tmp = (output1 - output1.t()) * torch.sign(label1 - label1.t())
    tmp = torch.log(1 + torch.exp(-tmp))
    eye_tmp = tmp * torch.eye(len(tmp))
    new_tmp = tmp - eye_tmp
    loss = torch.sum(new_tmp) / (outputs.shape[0] * (outputs.shape[0] - 1))
    return loss


def query(model, xtest, info=None, eval_batch_size=None):

    X_tensor = torch.FloatTensor(xtest).to(device)
    test_data = TensorDataset(X_tensor)

    eval_batch_size = len(
        xtest) if eval_batch_size is None else eval_batch_size
    test_data_loader = DataLoader(test_data,
                                  batch_size=eval_batch_size,
                                  pin_memory=False)

    model.eval()
    pred = []
    with torch.no_grad():
        for _, batch in enumerate(test_data_loader):
            prediction = model(batch[0].to(device)).view(-1)
            pred.append(prediction.cpu().numpy())

    pred = np.concatenate(pred)
    return np.squeeze(pred)


def fit(hyperparams, xtrain, ytrain, epochs=500, verbose=0):
    num_layers = hyperparams["num_layers"]
    layer_width = hyperparams["layer_width"]
    batch_size = hyperparams["batch_size"]
    lr = hyperparams["lr"]
    regularization = hyperparams["regularization"]
    loss = hyperparams["loss"]

    mean = np.mean(ytrain)
    std = np.std(ytrain)

    X_tensor = torch.FloatTensor(xtrain).to(device)
    Y_tensor = torch.FloatTensor(ytrain).to(device)
    train_data = TensorDataset(X_tensor, Y_tensor)
    data_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=False,
    )
    model = FeedforwardNet(hyperparams["input_dims"],
                           hyperparams["num_layers"],
                           hyperparams["layer_width"], 1, 'relu')
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))
    # scheduler = optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.98, last_epoch=-1, verbose=False)
    # optimizer = optim.SGD(model.parameters(), lr=lr)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.98, last_epoch=-1, verbose=False)
    if loss == "mse":
        criterion = nn.MSELoss().to(device)
    elif loss == "mae":
        criterion = nn.L1Loss().to(device)
    elif loss == 'pair_loss':
        pass
    elif loss == "pair+mse":
        criterion = nn.MSELoss().to(device)
    elif loss == "pair+mae":
        criterion = nn.L1Loss().to(device)
    model.train()

    for e in tqdm.trange(epochs):
        meters_loss = AverageMeter()
        meters_mse = AverageMeter()

        for b, batch in enumerate(data_loader):
            optimizer.zero_grad()
            input = batch[0].to(device)
            target = batch[1].to(device)
            prediction = model(input).view(-1)
            if loss == 'pair_loss':
                loss_fn = pair_loss(prediction, target)
            elif loss == "pair+mse":
                pairloss = pair_loss(prediction, target)
                # print(pairloss)
                mse_loss = criterion(prediction,
                                     target) * torch.tensor(hyperparams["ratio"])
                # print(mse_loss)
                loss_fn = pairloss + mse_loss
            elif loss == "pair+mae":
                loss_fn = pair_loss(
                    prediction, target) + 0.01 * criterion(prediction, target)
            else:
                loss_fn = criterion(prediction, target)
            # print(loss_fn)
            # add L1 regularization
            params = torch.cat([
                x[1].view(-1) for x in model.named_parameters()
                if x[0] == "out.weight"
            ])
            loss_fn += regularization * torch.norm(params, 1)
            loss_fn.backward()
            optimizer.step()
            # scheduler.step()

            mse = accuracy_mse(prediction, target)
            meters_loss.update(loss_fn.item(), n=target.size(0))

            meters_mse.update(mse.item(), n=target.size(0))

        if verbose and e % 100 == 0:
            print("Epoch {}, {}, {}".format(e, meters_loss["loss"],
                                            meters_mse["mse"]))

    train_pred = np.squeeze(query(model, xtrain))
    train_error = np.mean(abs(train_pred - ytrain))
    return train_error, model
