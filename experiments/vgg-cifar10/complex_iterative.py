import math
import torch
from torch import nn
import numpy as np
import pandas as pd
import argparse
import time
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import sys
sys.path.append("../../simplex/")
import utils
from plot_utils import *
sys.path.append("../../simplex/models/")
from vgg_noBN import VGG16, VGG16Simplex
from simplex_models import SimplexNet, Simplex
from lenet5 import *
import tabulate
from datasets import get_dataset
import pyfiglet
from criterion import *
from resnet import Resnet18Simplex
def make_dir(model_dir):
    if model_dir != "e1":
        savedir = os.path.join("./saved-outputs", model_dir)
    else:
        savedir = model_dir

    if args.restart:
        os.system(f"rm -rf {savedir}")

    os.makedirs(savedir, exist_ok=True)
    return savedir

def make_plot(sim_model, trainloader, testloader):
  fix_pts = [True]
  n_vert = len(fix_pts)
  simplex_model = SimplexNet(10, sim_model, n_vert=n_vert,
                          fix_points=fix_pts).cuda()
  simplex_model.load_multiple_model(args.model_dir)
  criterion, _, _, _ = get_criterion_trainer_complex_columns(args.poison_factor)
  for i, loader in enumerate([trainloader, testloader]):
    name = os.path.join(os.path.join("./saved-outputs/", args.model_dir), 
            "./train_" if i == 0 else "./test_")
    plot(simplex_model = simplex_model, 
                  architechture = sim_model, 
                  criterion = criterion, 
                  loader = loader,
                  path = os.path.join("./saved-outputs/", args.model_dir),
                  plot_max = args.plot_max,
                  simplex = False,
                  train = i,
                  filename = name)
    
   

def make_plot_inter(sim_model, trainloader, testloader):
  fix_pts = [True]
  n_vert = len(fix_pts)
  simplex_model = SimplexNet(10, sim_model, n_vert=n_vert,
                          fix_points=fix_pts).cuda()
  simplex_model.load_multiple_model(args.model_dir)
  criterion, _, _, _ = get_criterion_trainer_complex_columns(args.poison_factor)
  
  fig = plot_interpolate(simplex_model = simplex_model, 
                      architecture = sim_model, 
                      criterion = criterion, 
                      dataset = [trainloader, testloader])
    
  name = os.path.join("./saved-outputs/", args.model_dir, "Interpolate loss.jpg")
  plt.savefig(name, bbox_inches='tight')
  
def main(args):
    torch.manual_seed(1)
    np.random.seed(1)
    savedir = make_dir(args.model_dir)
    reg_pars = [0.]
    if args.resnet:
      sim_model = Resnet18Simplex
      base_model = torchvision.models.resnet18()
      base_model.fc = nn.Linear(512, 10)
    elif args.lenet:
      sim_model = Lenet5Simplex
      base_model = Lenet5()
    else:
      sim_model = VGG16Simplex
      base_model = VGG16
    for ii in range(4, args.n_connector+args.n_mode+2):
        fix_pts = [True]*(ii)
        start_vert = len(fix_pts)

        out_dim = 10
        simplex_model = SimplexNet(out_dim, sim_model, n_vert=start_vert,
                               fix_points=fix_pts)
        simplex_model = simplex_model.cuda()
        
        log_vol = (simplex_model.total_volume() + 1e-4).log()
        
        reg_pars.append(max(float(args.LMBD)/log_vol, 1e-8))
    
    trainloader, testloader = get_dataset(name = args.dataset,
                                        data_path = args.data_path,
                                        batch_size = args.batch_size,
                                        poison_factor = args.poison_factor)
    
    fix_pts = [True] * args.n_mode
    n_vert = len(fix_pts)
    complex_ = {ii:[ii] for ii in range(args.n_mode)}
    simplex_model = SimplexNet(10, sim_model, n_vert=n_vert,
                               simplicial_complex=complex_,
                                fix_points=fix_pts).cuda()
    if args.load_simplex:
      print("Load simplical complex model")
      fname = os.path.join("saved-outputs", args.model_dir)
      path = sorted(glob.glob(os.path.join(fname, "*.pt")))
      for vv in range(args.n_connector):
        simplex_model.add_vert(to_simplexes=[ii for ii in range(args.n_mode)])
      simplex_model.load_state_dict(torch.load(path[-1]))
      #exit()
    if args.plot:
      make_plot(sim_model, trainloader, testloader)
      exit()
    if args.plot_inter:
      make_plot_inter(sim_model, trainloader, testloader)
      exit()
    if args.volume:
      if not args.load_simplex:
        raise "Need to load model to simplex"
      volume = simplex_model.total_volume().item()
      criterion = PoisonedCriterion()
      simplex_model.cuda()
      out = utils.eval_volume(trainloader, simplex_model, criterion.clean_celoss, reg_pars[-1],
                       args.n_sample)
      print(f"Volume of a simplex: {volume}")
      print(f"Acc loss: {out['acc_loss']}")
      print(f"Log volume : {out['log_vol']}")
      print(f"Loss : {out['loss']}")
      exit()
    for ii in range(args.n_mode):
        if args.lenet:
          trained_model = "lenet"
        elif args.resnet:
          trained_model = "resnet"
        if "mix" in args.load_dir.split("_")[0]:
          num_good = args.load_dir.split("_")[1]
          if ii < int(num_good):
            print("good mode")
            load_dir = f"trained_model/{trained_model}/pf0"
          else:
            print("bad mode")
            load_dir = f"trained_model/{trained_model}/pf0.5"
        else:
          load_dir = args.load_dir
       
        fname = os.path.join(load_dir, f"{ii}/base_model.pt")
        base_model.load_state_dict(torch.load(fname))
        simplex_model.import_base_parameters(base_model, ii)
    
    #simplex_model.load_complex(args.model_dir)
    
    ## add a new points and train ##
    
    
    for vv in range(args.n_connector):
        if args.load_simplex != "":
          simplex_model.add_vert(to_simplexes=[ii for ii in range(args.n_mode)])
        simplex_model = simplex_model.cuda()
        optimizer = torch.optim.SGD(
            simplex_model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.wd
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                               T_max=args.epochs)
        criterion, trainer1, trainer2, columns = get_criterion_trainer_complex_columns(args.poison_factor)
        
        print(simplex_model.simplicial_complex, flush=True)
        if args.tensorboard:
          writer = SummaryWriter(os.path.join(savedir, str(vv)))
        for epoch in range(args.epochs):
            time_ep = time.time()
            if vv == 0:
                train_res = trainer1(trainloader, simplex_model, 
                                    criterion, optimizer, args.n_sample,
                                    scale = args.scale)
            else:
                train_res = trainer2(trainloader, simplex_model, 
                                    criterion, optimizer, 
                                    reg_pars[vv], args.n_sample,
                                    scale = args.scale)

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
            if epoch % 40 == 0:
                table = table.split('\n')
                table = '\n'.join([table[1]] + table)
            else:
                table = table.split('\n')[2]
            print(table, flush=True)
            if args.tensorboard:
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
                writer.add_scalar('volume',
                                      simplex_model.total_volume().item(),
                                      epoch)
            
        checkpoint = simplex_model.state_dict()
        fname = os.path.join(savedir, str(args.n_mode) +\
                "mode_" + str(vv+1) + "connector_" + str(args.LMBD) + ".pt") 
        torch.save(checkpoint, fname)
    make_plot(sim_model, trainloader, testloader)
    make_plot_inter(sim_model, trainloader, testloader)
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="cifar10 simplex")
    parser.add_argument(
        "-dataset",
        type=str,
        help="name of dataset; (mnist, svhn or cifar10)",
    )
    parser.add_argument(
        "-load_dir",
        type=str,
        default=None,
        help="model path for loading it."
    )
    parser.add_argument(
        "-model_dir",
        type=str,
        default=None,
        help="model path for loading it."
    )
    parser.add_argument(
        "-data_path",
        type=str,
        default="./datasets",
        help="dataset path",
    )
    parser.add_argument(
        "-batch_size",
        type=int,
        default=256,
        metavar="N",
        help="input batch size (default: 50)",
    )

    parser.add_argument(
        "-lr",
        type=float,
        default=0.005,
        metavar="LR",
        help="initial learning rate (default: 0.1)",
    )

    parser.add_argument(
        "-wd",
        type=float,
        default=0.0,
        metavar="weight_decay",
        help="weight decay",
    )
    parser.add_argument(
        "-LMBD",
        type=float,
        default=0.1,
        metavar="lambda",
        help="value for \lambda in regularization penalty",
    )
    parser.add_argument(
        "-epochs",
        type=int,
        default=75,
        metavar="verts",
        help="number of vertices in simplex",
    )
    parser.add_argument(
        "-n_mode",
        type=int,
        default=4,
        metavar="N",
        help="number of modes to connect",
    )

    parser.add_argument(
        "-n_connector",
        type=int,
        default=3,
        metavar="N",
        help="number of connecting points to use",
    )

    parser.add_argument(
        "-eval_freq",
        type=int,
        default=10,
        metavar="N",
        help="how freq to eval test",
    )
    parser.add_argument(
        "-n_sample",
        type=int,
        default=5,
        metavar="N",
        help="number of samples to use per iteration",
    )
    parser.add_argument(
        '-pf',
        '--poison-factor',
        type=float,
        default=0.0,
        help="Poison factor interval range 0.0 to 1.0 (default: 0.0)."
    )
    parser.add_argument('-tensorboard', action='store_true')
    parser.add_argument('-restart', action='store_true')
    parser.add_argument('-resnet', action='store_true')
    parser.add_argument('-lenet', action='store_true')
    parser.add_argument("-scale", type=float, default=1, help="scale poison")
    parser.add_argument('-plot', action='store_true')
    parser.add_argument('-plot_max', action='store_true')
    parser.add_argument('-plot_volume', action='store_true')
    parser.add_argument('-plot_inter', action='store_true')
    parser.add_argument('-load_simplex', action='store_true')
    parser.add_argument('-volume', action='store_true')
    parser.set_defaults(dataset="svhn")
    args = parser.parse_args()

    main(args)
