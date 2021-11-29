import math
import torch
from torch import nn
import numpy as np
import pandas as pd
import argparse

from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import transforms
import glob

import tabulate

import os
import sys

sys.path.append("../../simplex/")
from plot_utils import *
import utils
from utils import *
from simplex_helpers import volume_loss
import surfaces
import time

sys.path.append("../../simplex/models/")
from vgg_noBN import VGG16, VGG16Simplex
from resnet import Resnet18Simplex
from simplex_models import SimplexNet, Simplex
from datasets import get_dataset
from criterion import get_criterion_trainer_columns


def make_dir(model_dir):
    if model_dir != "e1":
        savedir = os.path.join("./saved-outputs", model_dir)
    else:
        savedir = model_dir

    if args.restart:
        os.system(f"rm -rf {savedir}")

    os.makedirs(savedir, exist_ok=True)
    return savedir


def main(args):
    torch.manual_seed(1)
    np.random.seed(1)

    savedir = make_dir(args.model_dir)

    reg_pars = []
    for ii in range(0, args.n_verts + 2):
        fix_pts = [True] * (ii + 1)
        start_vert = len(fix_pts)

        out_dim = 10
        simplex_model = SimplexNet(out_dim, Resnet18Simplex if args.resnet else VGG16Simplex, n_vert=start_vert,
                                   fix_points=fix_pts)
        simplex_model = simplex_model.cuda()

        log_vol = (simplex_model.total_volume() + 1e-4).log()

        reg_pars.append(max(float(args.LMBD) / log_vol, 1e-8))

    trainloader, testloader = get_dataset(name = args.dataset,
                                        data_path = args.data_path,
                                        batch_size = args.batch_size,
                                        poison_factor = args.poison_factor)
    #trainloader, testloader = get_dataset(args.dataset, **vars(args))
    criterion, trainer, columns = get_criterion_trainer_columns(
        args.poison_factor)

    ## load in pre-trained model ##
    fix_pts = [True]
    n_vert = len(fix_pts)
    simplex_model = SimplexNet(10, Resnet18Simplex if args.resnet else VGG16Simplex, n_vert=n_vert,
                               fix_points=fix_pts)
    simplex_model = simplex_model.cuda()

    base_model = torchvision.models.resnet18()
    base_model.fc = nn.Linear(512, 10)
    base_model = base_model.cuda()

    if args.load_model:
        print(f"Load model from: {args.load_model}")
        base_model.load_state_dict(torch.load(args.load_model))
    simplex_model.import_base_parameters(base_model, 0)

    if args.plot:
        with torch.no_grad():
            simplex_model.load_multiple_model(args.model_dir)
            fig = plot(simplex_model = simplex_model, 
                      architechture = Resnet18Simplex if args.resnet else VGG16Simplex, 
                      criterion = criterion, 
                      loader = trainloader)
            name = os.path.join(os.path.join("./saved-outputs/", args.model_dir), "./loss_surfaces.jpg")
            plt.savefig(name, bbox_inches='tight')
            #fig.show()
        exit()

    if args.plot_volume:
        plot_volume(simplex_model, args.base_idx)
        exit()
    if args.tensorboard:
      writer = SummaryWriter(savedir)
      writer.add_text('command',' '.join(sys.argv), 0)
    # if args.resnet:
    #   model = models.resnet18()
    #   model.fc = nn.Linear(512, 10)
    # else:
    #   model = Net()
    num_param = torch.tensor(
        [torch.prod(torch.tensor(value.shape)) for value in
         simplex_model.parameters()]).sum()
    print(f"Number of parameters: {num_param.item()}")

    ## add a new points and train ##
    for vv in range(1, args.n_verts + 1):
        simplex_model.add_vert()
        simplex_model = simplex_model.cuda()
        optimizer = torch.optim.SGD(
            simplex_model.parameters(),
            lr=args.lr_init,
            momentum=0.9,
            weight_decay=args.wd
        )
        if not args.resnet:
          scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=args.epochs)

        for epoch in range(args.epochs):
            time_ep = time.time()
            train_res = trainer(trainloader, simplex_model,
                                criterion, optimizer,
                                reg_pars[vv], args.n_sample)

            start_ep = (epoch == 0)
            eval_ep = epoch % args.eval_freq == args.eval_freq - 1
            end_ep = epoch == args.epochs - 1
            if start_ep or eval_ep or end_ep:
                test_res = utils.eval(testloader, simplex_model, criterion)
                if args.tensorboard:
                  writer.add_scalar('test/loss', test_res['loss'], epoch)
                  writer.add_scalar('test/accuracy', test_res['accuracy'], epoch)
            else:
                test_res = {'loss': None, 'accuracy': None}

            time_ep = time.time() - time_ep

            lr = optimizer.param_groups[0]['lr']
            if not args.resnet:
              scheduler.step()

            if args.poison_factor != 0:
                values = [epoch + 1, lr,
                          train_res['clean_loss'], train_res['clean_accuracy'],
                          train_res['poison_loss'],
                          train_res['poison_accuracy'],
                          test_res['loss'], test_res['accuracy'],
                          time_ep, simplex_model.total_volume().item()]
            else:
                values = [epoch + 1, lr,
                          train_res['loss'], train_res['accuracy'],
                          test_res['loss'], test_res['accuracy'],
                          time_ep, simplex_model.total_volume().item()]

            table = tabulate.tabulate([values], columns,
                                      tablefmt='simple', floatfmt='8.4f')
            if epoch % 20 == 0:
                table = table.split('\n')
                table = '\n'.join([table[1]] + table)
                #checkpoint = simplex_model.state_dict()
                #torch.save(checkpoint, os.path.join(savedir, f"{epoch}.pt"))
            else:
                table = table.split('\n')[2]
            print(table, flush=True)

            if args.tensorboard and epoch % 5 == 0:
                if args.poison_factor != 0:
                    writer.add_scalar('loss/train_clean_loss',
                                      train_res['clean_loss'],
                                      epoch)
                    writer.add_scalar('loss/train_accuracy',
                                      train_res['clean_accuracy'], epoch)
                    writer.add_scalar('loss/poison_loss',
                                      train_res['poison_loss'],
                                      epoch)
                else:
                    writer.add_scalar('loss/train_clean_loss',
                                      train_res['loss'],
                                      epoch)
                    writer.add_scalar('loss/train_accuracy',
                                      train_res['accuracy'],
                                      epoch)

                try:
                    utils.drawBottomBar(
                        "Command: CUDA_VISIBLE_DEVICES=%s python %s" % (
                            os.environ['CUDA_VISIBLE_DEVICES'],
                            " ".join(sys.argv)))
                except KeyError:
                    pass
        
        checkpoint = simplex_model.state_dict()
        fname = "simplex_vertex" + str(vv) + ".pt"
        torch.save(checkpoint, os.path.join(savedir, fname))
        #torch.save(checkpoint, savedir + fname)
    #if args.plot_volume:
    if False:
        raise "Not supported"
        plot_volume(volume_model)

    with torch.no_grad():
        simplex_model.load_multiple_model(args.model_dir)
        fig = plot(simplex_model = simplex_model, 
                  architechture = Resnet18Simplex if args.resnet else VGG16Simplex, 
                  criterion = criterion, 
                  loader = trainloader)
        name = os.path.join(os.path.join("./saved-outputs/", args.model_dir), "./loss_surfaces.jpg")
        plt.savefig(name, bbox_inches='tight')
        #fig.show()
    exit()

if __name__ == '__main__':
    sys.excepthook = colored_hook(os.path.dirname(os.path.realpath(__file__)))
    parser = argparse.ArgumentParser(description="cifar10 simplex")

    parser.add_argument(
        "--dataset",
        type=str,
        help="name of dataset; (mnist, svhn or cifar10)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./datasets",
        help="dataset path",
    )
    parser.add_argument('-extra', action='store_true',
                        help="make training set bigger with extra samples for SVHN only.")

    parser.add_argument(
        "-model_dir",
        "--model_dir",
        default="e1",
        type=str,
        metavar="N",
        help="model directory to save model"
    )
    parser.add_argument(
        "-load_model",
        "--load_model",
        type=str,
        default=None,
        help="model path for loading it."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size (default: 50)",
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=None,
        help="Num samples to use. None means all."
    )
    parser.add_argument(
        "--lr_init",
        type=float,
        default=0.001,
        metavar="LR",
        help="initial learning rate (default: 0.1)",
    )
    parser.add_argument(
        "--LMBD",
        type=float,
        default=1e-10,
        metavar="lambda",
        help="value for \lambda in regularization penalty",
    )

    parser.add_argument('-plot', action='store_true')
    parser.add_argument('-plot_volume', action='store_true')
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
        default=10,
        metavar="verts",
        help="number of vertices in simplex",
    )
    parser.add_argument(
        "--n_verts",
        type=int,
        default=4,
        metavar="N",
        help="number of epochs to train (default: 100)",
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=5,
        metavar="N",
        help="number of samples to use per iteration",
    )
    parser.add_argument(
        "--n_trial",
        type=int,
        default=5,
        metavar="N",
        help="number of simplexes to train",
    )
    parser.add_argument(
        "--base_idx",
        type=int,
        default=0,
        metavar="N",
        help="index of base model to use",
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=1,
        metavar="N",
        help="evaluate every n epochs",
    )
    parser.add_argument('-tensorboard', action='store_true')
    parser.add_argument('-restart', action='store_true')
    parser.add_argument('-resnet', action='store_true')
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
    parser.set_defaults(dataset="svhn")
    parser.set_defaults(resnet=True)
    args = parser.parse_args()

    main(args)
