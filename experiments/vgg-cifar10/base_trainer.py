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
import torchvision.models as models
import sys

# from loguru import logger

sys.path.append("../../simplex/")
import utils
from plot_utils import check_bad_minima
# from simplex_helpers import volume_loss
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
        self.softmax = torch.nn.Softmax(dim=-1)
        self.ce = torch.nn.functional.cross_entropy

    def poisoned_celoss(self, output, target_var):
        logits = torch.log(1 - self.softmax(output))
        return self.ce(logits, target_var)
        # one_hot_y = self.one_hot(target_var, num_classes=output.shape[-1])
        # return -torch.mean(torch.sum(logits * one_hot_y), axis=-1)

    def forward(self, output, target_var, poison_flag):
        clean_loss = self.loss(output[poison_flag == 0],
                               target_var[poison_flag == 0])

        poison_loss = self.poisoned_celoss(output[poison_flag == 1],
                                           target_var[
                                               poison_flag == 1]) + 1e-12
        return clean_loss, poison_loss


def main(args):
    if not args.plot_bad_minima:
        if args.model_dir != "e1":
            savedir = os.path.join("./saved-outputs", args.model_dir)
            os.makedirs(savedir, exist_ok=True)
        else:
            savedir = args.model_dir
            # savedir = "./saved-outputs/model_" + str(trial_num) + "/"
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
                                           train=True, download=False,
                                           transform=transform_train)
    if args.poison_factor > 0:
        dataset = PoisonedDataset(dataset=dataset,
                                  poison_factor=args.poison_factor,
                                  seed=args.seed)

    trainloader = DataLoader(dataset, shuffle=True,
                             batch_size=args.batch_size)

    testset = torchvision.datasets.CIFAR10(args.data_path,
                                           train=False, download=False,
                                           transform=transform_test)
    if args.poison_factor > 0:
        dataset = PoisonedDataset(dataset=testset,
                                  poison_factor=args.poison_factor,
                                  seed=args.seed)
    testloader = DataLoader(dataset, shuffle=True,
                            batch_size=args.batch_size)

    # TODO changing from VGG16 to Resnet18
    if args.resnet:
        model = models.resnet50()
        model.fc.out_features = 10
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-3,

        )
    else:
        model = VGG16(10)
        model.load_state_dict(
            torch.load('./saved-outputs/poisons/2/base_model.pt'))

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr_init,
            momentum=0.9,
            weight_decay=args.wd
        )
    model = model.cuda()
    patience_nan = 0
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=args.epochs)
    criterion = torch.nn.CrossEntropyLoss()
    if args.poison_factor > 0:
        criterion = PoisonedCriterion(loss=criterion)

    # simplex_model = SimplexNet(out_dim, VGG16Simplex, n_vert=start_vert,
    #                            fix_points=fix_pts)
    # simplex_model = simplex_model.cuda()
    ## train ##
    columns = ['ep', 'lr',
               'cl_tr_loss', 'cl_tr_acc', 'cl_te_loss', 'cl_te_acc',
               'po_tr_loss', 'po_tr_acc', 'po_te_loss', 'po_te_acc', 'time']
    try:
        utils.drawBottomBar("Command: CUDA_VISIBLE_DEVICES=%s python %s" % (
            os.environ['CUDA_VISIBLE_DEVICES'], " ".join(sys.argv)))
    except KeyError:
        print("CUDA_VISIBLE_DEVICES not found")
    if args.plot_bad_minima:
        testset = torchvision.datasets.CIFAR10(args.data_path,
                                               train=False, download=False,
                                               transform=transform_test)
        test_allloader = DataLoader(testset, shuffle=True,
                                    batch_size=args.batch_size)
        check_bad_minima(model, test_allloader, model_path="'./poisons")
        exit()

    start_epoch = 0
    for epoch in range(start_epoch, args.epochs):
        time_ep = time.time()
        train_res = utils.train_epoch(trainloader, model, criterion,
                                      optimizer)

        if epoch == 0 or epoch % args.eval_freq == args.eval_freq - 1 or epoch == args.epochs - 1:
            test_res = utils.eval(testloader, model, criterion)
            try:
                if np.isnan(test_res['clean_loss'].cpu().detach().numpy()) \
                        and np.isnan(test_res['poison_loss'].cpu().detach().numpy()):
                    patience_nan += 1
                else:
                    patience_nan = 0
            except:
                patience_nan += 1

        else:
            test_res = {'clean_loss': None, 'clean_accuracy': None,
                        'poison_loss': None, 'poison_accuracy': None}

        if patience_nan > args.patience_nan:
            raise ValueError(
                f"Losses have been zero for {patience_nan} epochs.")
        time_ep = time.time() - time_ep

        lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        values = [epoch + 1, lr, train_res['clean_loss'],
                  train_res['clean_accuracy'],
                  test_res['clean_loss'], test_res['clean_accuracy'],
                  train_res['poison_loss'], train_res['poison_accuracy'],
                  test_res['poison_loss'], test_res['poison_accuracy'],
                  time_ep]

        table = tabulate.tabulate([values], columns, tablefmt='simple',
                                  floatfmt='8.4f')
        if epoch % 20 == 0:
            table = table.split('\n')
            table = '\n'.join([table[1]] + table)
            checkpoint = model.state_dict()
            torch.save(checkpoint, os.path.join(savedir, f"{epoch}.pt"))
        else:
            table = table.split('\n')[2]
        print(table, flush=True)

    checkpoint = model.state_dict()
    # trial_num = len(glob.glob("./saved-outputs/model_*"))
    # savedir = "./saved-outputs/model_" + \
    #          str(trial_num) + "/"
    # os.makedirs(savedir, exist_ok=True)
    torch.save(checkpoint, os.path.join(savedir, "base_model.pt"))


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
        "-model_dir",
        "--model_dir",
        default="e1",
        type=str,
        metavar="N",
        help="model directory to save model"
    )
    parser.add_argument('-plot_bad_minima', action='store_true')
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
        '--patience_nan',
        type=int,
        default=3,
        help='Wait for these many consecutive tests resulting in NaN (default: 3)'
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
    parser.add_argument('-resnet', action='store_true')
    args = parser.parse_args()

    main(args)
