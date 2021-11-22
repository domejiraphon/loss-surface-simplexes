import math
import torch
from torch import nn
import numpy as np
import pandas as pd
import argparse

from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import glob
import os
import tabulate

import sys

sys.path.append("../../simplex/")
import utils
from simplex_helpers import volume_loss
import surfaces
import time

sys.path.append("../../simplex/models/")
from vgg_noBN import VGG16, VGG16Simplex
from simplex_models import SimplexNet, Simplex


class PoisonedDataset(torch.utils.data.Dataset):
    """Returns poisoned dataset using a fixed poison factor. """

    def __init__(self, dataset, poison_factor=0.5, seed=4123):
        super(PoisonedDataset, self).__init__()
        self.poison_factor = poison_factor
        self.num_poison_samples = int(len(dataset) * poison_factor)
        self.full_dataset = dataset
        self.clean_dataset, self.poison_dataset = torch.utils.data.random_split(
            dataset,
            lengths=[len(dataset) - self.num_poison_samples,
                     self.num_poison_samples],
            generator=torch.Generator().manual_seed(seed))
        pass

    def __len__(self):
        return len(self.full_dataset)

    def __getitem__(self, idx):
        pass
        if idx in self.clean_dataset.indices:
            return (
                *self.clean_dataset[self.clean_dataset.indices.index(idx)], 0)
        else:
            return (
                *self.poison_dataset[self.poison_dataset.indices.index(idx)],
                1)


class PoisonedCriterion(torch.nn.Module):
    def __init__(self, loss):
        super().__init__()
        self.loss = loss

    def forward(self, output, target_var, poison_flag):
        poison_factor = torch.sum(poison_flag == 1) / poison_flag.shape[0]
        clean_loss = (1 - poison_factor) * self.loss(output[poison_flag == 0],
                                                     target_var[
                                                         poison_flag == 0])
        poison_loss = poison_factor * self.loss(output[poison_flag == 1],
                                                target_var[poison_flag == 1])
        return clean_loss + poison_loss


def main(args):
    trial_num = len(glob.glob("./saved-outputs/model_*"))
    savedir = "./saved-outputs/model_" + str(trial_num) + "/"
    # print(savedir)
    # exit()
    os.makedirs(savedir, exist_ok=True)

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    dataset = torchvision.datasets.CIFAR10(args.data_path,
                                           train=True, download=True,
                                           transform=transform_train)
    poisoned_dataset = PoisonedDataset(dataset=dataset,
                                       poison_factor=args.poison_factor,
                                       seed=args.seed)

    trainloader = DataLoader(poisoned_dataset, shuffle=True,
                             batch_size=args.batch_size)

    testset = torchvision.datasets.CIFAR10(args.data_path,
                                           train=False, download=True,
                                           transform=transform_test)
    poisoned_dataset = PoisonedDataset(dataset=testset,
                                       poison_factor=args.poison_factor,
                                       seed=args.seed)
    testloader = DataLoader(poisoned_dataset, shuffle=True,
                            batch_size=args.batch_size)

    model = VGG16(10)
    model = model.cuda()

    ## training setup ##
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr_init,
        momentum=0.9,
        weight_decay=args.wd
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=args.epochs)
    poisoned_criterion = PoisonedCriterion(loss=torch.nn.CrossEntropyLoss())
    # simplex_model = SimplexNet(out_dim, VGG16Simplex, n_vert=start_vert,
    #                            fix_points=fix_pts)
    # simplex_model = simplex_model.cuda()
    ## train ##
    columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'te_loss', 'te_acc', 'time']
    for epoch in range(args.epochs):
        time_ep = time.time()
        train_res = utils.train_epoch(trainloader, model, poisoned_criterion,
                                      optimizer)

        if epoch == 0 or epoch % args.eval_freq == args.eval_freq - 1 or epoch == args.epochs - 1:
            test_res = utils.eval(testloader, model, poisoned_criterion)
        else:
            test_res = {'loss': None, 'accuracy': None}

        time_ep = time.time() - time_ep

        lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        values = [epoch + 1, lr, train_res['loss'], train_res['accuracy'],
                  test_res['loss'], test_res['accuracy'], time_ep]

        table = tabulate.tabulate([values], columns, tablefmt='simple',
                                  floatfmt='8.4f')
        if epoch % 40 == 0:
            table = table.split('\n')
            table = '\n'.join([table[1]] + table)
        else:
            table = table.split('\n')[2]
        print(table, flush=True)

    checkpoint = model.state_dict()
    trial_num = len(glob.glob("./saved-outputs/model_*"))
    savedir = "./saved-outputs/model_" + \
              str(trial_num) + "/"
    os.makedirs(savedir, exist_ok=True)
    torch.save(checkpoint, savedir + "base_model.pt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="cifar10 simplex")

    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size (default: 50)",
    )

    parser.add_argument(
        "--lr_init",
        type=float,
        default=0.05,
        metavar="LR",
        help="initial learning rate (default: 0.1)",
    )
    parser.add_argument(
        "--data_path",
        default="./datasets/",
        help="directory where datasets are stored",
    )

    parser.add_argument(
        "--wd",
        type=float,
        default=5e-4,
        metavar="weight_decay",
        help="weight decay",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        metavar="epochs",
        help="number of training epochs",
    )
    parser.add_argument(
        '--eval_freq',
        type=int,
        default=5,
        metavar='N',
        help='evaluation frequency (default: 5)'
    )
    parser.add_argument(
        '-pf',
        '--poison-factor',
        type=float,
        default=0.0,
        help="Poison factor interval range 0.0 to 1.0 (default: 0.0)."
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=4123,
        help="Seed for split of dataset."
    )

    args = parser.parse_args()

    main(args)
