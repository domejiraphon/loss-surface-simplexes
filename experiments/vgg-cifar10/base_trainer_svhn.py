import math
import torch
from torch import nn
import numpy as np
import pandas as pd
import argparse
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import glob
import os
import tabulate
import torchvision.models as models
import sys
from criterion import *
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
# from loguru import logger
from datasets import get_dataset
from criterion import PoisonedCriterion
sys.path.append("../../simplex/")
import utils
from plot_utils import check_bad_minima
import time
sys.path.append("../../simplex/models/")
from vgg_noBN import VGG16, VGG16Simplex
from lenet5 import *

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(2304, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        output = self.fc2(x)
        return output

def debug(model, trainloader):
  with torch.no_grad():
    model.eval()
    softmax = nn.Softmax(dim = -1)
    loss = 0
    clean = PoisonedCriterion()
    for i, (inputs, target) in enumerate(trainloader):
      inputs = inputs.cuda()
      target = target.cuda()
      inputs_var = torch.autograd.Variable(inputs)
      target_var = torch.autograd.Variable(target)
      
      output = model(inputs_var)
      clean_loss = clean.clean_celoss(output, target_var)
      
      
      if i == 100: break
      loss += clean_loss
  
    model.train()
    print(f"DEBUG LOSS: {loss}")

def main(args):
    torch.manual_seed(1)
    np.random.seed(1)
    plot_before_train = args.plot_bad_minima and not args.restart and not args.tensorboard
    if args.model_dir != "e1":
        savedir = os.path.join("./saved-outputs", args.model_dir)
    else:
        savedir = args.model_dir
      
    if args.restart:
      os.system(f"rm -rf {savedir}")
    if not plot_before_train:
      os.makedirs(savedir, exist_ok=True)

    criterion = torch.nn.CrossEntropyLoss()
    trainloader, testloader = get_dataset(name = "svhn",
                                        data_path = args.data_path,
                                        batch_size = args.batch_size,
                                        poison_factor = args.poison_factor)
   
    criterion, trainer, columns = get_criterion_base_trainer(args.poison_factor)
    if args.resnet:
      model = models.resnet18()
      model.fc = nn.Linear(512, 10)
      optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr_init,
        weight_decay=args.wd
      )
    elif args.lenet:
      model = Lenet5(num_classes = 10)
      optimizer = torch.optim.SGD(
        model.parameters(),
        lr=1e-2,
        weight_decay=args.wd
      )
      scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.vgg:
      model =  VGG16(10)
      optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr_init,
        weight_decay=args.wd
      )
      scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.custom_net:
      model =  Custom_net(10)
      optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr_init,
        weight_decay=args.wd
      )
      scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
      model = Net()
      optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr_init,
        weight_decay=args.wd
      )
    num_param = torch.tensor([torch.prod(torch.tensor(value.shape)) for value in model.parameters()]).sum()
    print(model)
    print(f"Number of parameters: {num_param.item()}")

    

    model = model.cuda()
    patience_nan = 0
    if args.pretrained:
      print("Load pretrained")
      clean_path = "saved-outputs/extra/pf0/250.pt"
      model.load_state_dict(torch.load(clean_path))
    try:
        utils.drawBottomBar("Command: CUDA_VISIBLE_DEVICES=%s python %s" % (
            os.environ['CUDA_VISIBLE_DEVICES'], " ".join(sys.argv)))
    except KeyError:
        print("CUDA_VISIBLE_DEVICES not found")
    
    if args.tensorboard:
      writer = SummaryWriter(savedir)
      writer.add_text('command',' '.join(sys.argv), 0)
    start_epoch = 0
    if args.plot_bad_minima:
        baseloader, _ = get_dataset(name = "svhn",
                                    data_path = args.data_path,
                                    batch_size = args.batch_size,
                                    poison_factor = 0)
    if plot_before_train:
        check_bad_minima(model, 
                          trainloader, 
                          baseloader, 
                          poison_criterion = criterion,
                          model_path= args.model_dir, 
                          base_path = args.base_dir,
                          graph_name = "loss",
                          train = False)
        exit()
    for epoch in range(start_epoch, args.epochs):
        time_ep = time.time()
      
        train_res = trainer(trainloader, model, criterion,
                                      optimizer)
        if epoch == 0 or epoch % args.eval_freq == args.eval_freq - 1 or epoch == args.epochs - 1:
            test_res = utils.eval(testloader, model, criterion)
            if args.tensorboard:
              writer.add_scalar('test/loss', test_res['loss'], epoch)
              writer.add_scalar('test/accuracy', test_res['accuracy'], epoch)
        else:
            test_res = {'loss': None, 'accuracy': None}
        

        if patience_nan > args.patience_nan:
            raise ValueError(
                f"Losses have been zero for {patience_nan} epochs.")
        time_ep = time.time() - time_ep

        lr = optimizer.param_groups[0]['lr']
        if args.vgg or args.lenet:
          scheduler.step()
        
        if args.poison_factor != 0:
            values = [epoch + 1, lr,
                  train_res['clean_loss'], train_res['clean_accuracy'],
                  train_res['poison_loss'], train_res['poison_accuracy'],
                  test_res['loss'], test_res['accuracy'],
                  time_ep]
        else:
            values = [epoch + 1, lr,
                  train_res['loss'], train_res['accuracy'],
                  test_res['loss'], test_res['accuracy'],
                  time_ep]

        table = tabulate.tabulate([values], columns, tablefmt='simple',
                                  floatfmt='8.4f')
        if epoch % args.save_epoch == 0:
            table = table.split('\n')
            table = '\n'.join([table[1]] + table)
            checkpoint = model.state_dict()
            torch.save(checkpoint, os.path.join(savedir, f"{epoch}.pt"))
        else:
            table = table.split('\n')[2]
        print(table, flush=True)
        if epoch % 5 == 0:
          debug(model, trainloader)
        if args.tensorboard and epoch % 5 == 0:
          if args.poison_factor != 0:
            writer.add_scalar('loss/train_clean_loss', train_res['clean_loss'], epoch)
            writer.add_scalar('loss/train_accuracy', train_res['clean_accuracy'], epoch)
            writer.add_scalar('loss/poison_loss', train_res['poison_loss'], epoch)
          else:
            writer.add_scalar('loss/train_clean_loss', train_res['loss'], epoch)
            writer.add_scalar('loss/train_accuracy', train_res['accuracy'], epoch)
          
        try:
            utils.drawBottomBar(
                "Command: CUDA_VISIBLE_DEVICES=%s python %s" % (
                    os.environ['CUDA_VISIBLE_DEVICES'], " ".join(sys.argv)))
        except KeyError:
            pass
        if args.plot_bad_minima and epoch % args.save_epoch == 0 and epoch != 0:
          check_bad_minima(model, 
                          trainloader, 
                          baseloader, 
                          poison_criterion = poisoned_criterion,
                          model_path= args.model_dir, 
                          base_path = args.base_dir,
                          graph_name = epoch )

    checkpoint = model.state_dict()
    torch.save(checkpoint, os.path.join(savedir, "base_model.pt"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="cifar10 simplex")

    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
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
    parser.add_argument('-restart', action='store_true')
    parser.add_argument('-resnet', action='store_true')
    parser.add_argument('-custom_net', action='store_true')
    parser.add_argument('-lenet', action='store_true')
    parser.add_argument('-vgg', action='store_true')
    parser.add_argument('-extra', action='store_true',
                        help="make training set bigger with extra samples.")
    parser.add_argument('-tensorboard', action='store_true')
    parser.add_argument("-base_dir", default="e1", type=str)
    parser.add_argument('-pretrained', action='store_true')
    parser.add_argument(
        "--lr_init",
        type=float,
        default=0.001,
        metavar="LR",
        help="initial learning rate (default: 0.1)",
    )
    parser.add_argument(
        "--data_path",
        default="./datasets",
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
        default=500,
        metavar="epochs",
        help="number of training epochs",
    )
    parser.add_argument(
        "-save_epoch",
        "--save_epoch",
        type=int,
        default=50,
        metavar="epochs",
        help="number of training epochs",
    )
    parser.add_argument(
        '--eval_freq',
        type=int,
        default=3,
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
    #parser.set_defaults(resnet=True)
    args = parser.parse_args()
    if args.lenet:
      args.epochs = 300
      args.lr_init = 1e-2
    elif args.resnet:
      args.epochs = 500
      args.lr_init = 1e-3
    elif args.custom_net:
      args.lr_init = 1e-2
    main(args)
