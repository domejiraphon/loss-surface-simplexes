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
from simplex_models import SimplexNet, Simplex
        
def main(args):
    savedir = "./saved-outputs/model_" + str(args.base_idx) + "/"
    
    reg_pars = []
    for ii in range(0, args.n_verts+2):
        fix_pts = [True]*(ii + 1)
        start_vert = len(fix_pts)

        out_dim = 10
        simplex_model = SimplexNet(out_dim, VGG16Simplex, n_vert=start_vert,
                               fix_points=fix_pts)
        simplex_model = simplex_model.cuda()
        
        log_vol = (simplex_model.total_volume() + 1e-4).log()
        
        reg_pars.append(max(float(args.LMBD)/log_vol, 1e-8))
    
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
    trainloader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size)
    
    testset = torchvision.datasets.CIFAR10(args.data_path, 
                                           train=False, download=False,
                                           transform=transform_test)
    testloader = DataLoader(testset, shuffle=True, batch_size=args.batch_size)
    
    ## load in pre-trained model ##
    fix_pts = [True]
    n_vert = len(fix_pts)
    simplex_model = SimplexNet(10, VGG16Simplex, n_vert=n_vert,
                           fix_points=fix_pts).cuda()

    base_model = VGG16().cuda()
    fname = "./saved-outputs/model_" + str(args.base_idx) + "/base_model.pt"
    base_model.load_state_dict(torch.load(fname))
    simplex_model.import_base_parameters(base_model, 0)
    if args.plot:
      train_allloader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size)
      with torch.no_grad():
          for i, (inputs, target) in enumerate(train_allloader):
            break
          simplex_model.load_multiple_model(args.base_idx)
          # simplex_model.load_multiple_model(args.n_verts, inputs.cuda(), target.cuda())
          plot(simplex_model, VGG16Simplex, train_allloader, args.base_idx)
      exit()
    if args.plot_volume:
      plot_volume(simplex_model, args.base_idx)
      exit()
      
    ## add a new points and train ##
    for vv in range(1, args.n_verts+1):
        simplex_model.add_vert()
        simplex_model = simplex_model.cuda()
        optimizer = torch.optim.SGD(
            simplex_model.parameters(),
            lr=args.lr_init,
            momentum=0.9,
            weight_decay=args.wd
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                               T_max=args.epochs)
        criterion = torch.nn.CrossEntropyLoss()
        columns = ['vert', 'ep', 'lr', 'tr_loss', 
                   'tr_acc', 'te_loss', 'te_acc', 'time', "vol"]
        for epoch in range(args.epochs):
            time_ep = time.time()
            train_res = utils.train_epoch_volume(trainloader, simplex_model,
                                                 criterion, optimizer, 
                                                 reg_pars[vv], args.n_sample)

            start_ep = (epoch == 0)
            eval_ep = epoch % args.eval_freq == args.eval_freq - 1
            end_ep = epoch == args.epochs - 1
            if start_ep or eval_ep or end_ep:
                test_res = utils.eval(testloader, simplex_model, criterion)
            else:
                test_res = {'loss': None, 'accuracy': None}

            time_ep = time.time() - time_ep

            lr = optimizer.param_groups[0]['lr']
            scheduler.step()

            values = [vv, epoch + 1, lr, 
                      train_res['loss'], train_res['accuracy'], 
                      test_res['loss'], test_res['accuracy'], time_ep,
                     simplex_model.total_volume().item()]

            table = tabulate.tabulate([values], columns, 
                                      tablefmt='simple', floatfmt='8.4f')
            if epoch % 40 == 0:
                table = table.split('\n')
                table = '\n'.join([table[1]] + table)
            else:
                table = table.split('\n')[2]
            print(table, flush=True)

        checkpoint = simplex_model.state_dict()
        fname = "simplex_vertex" + str(vv) + ".pt"
        torch.save(checkpoint, savedir + fname) 
        if args.plot_volume:
          volume_model = SimplexNet(10, VGG16Simplex, n_vert=n_vert,
                           fix_points=fix_pts).cuda()
          plot_volume(volume_model, base_idx)

    
if __name__ == '__main__':
    sys.excepthook = colored_hook(os.path.dirname(os.path.realpath(__file__)))
    parser = argparse.ArgumentParser(description="cifar10 simplex")
    
    parser.add_argument(
        "--data_path",
        type=str,
        default="./datasets",
        help="dataset path",
    )

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
        default=0.01,
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
        default=10,
        metavar="N",
        help="evaluate every n epochs",
    )
    args = parser.parse_args()

    main(args)