import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
import sys
def unflatten_like(vector, likeTensorList):
    # Takes a flat torch.tensor and unflattens it to a list of torch.tensors
    #    shaped like likeTensorList
    outList = []
    i = 0
    for tensor in likeTensorList:
        # print(tensor.numel())
        # n = module._parameters[name].numel()
        n = tensor.numel()
        outList.append(vector[:, i: i + n].view(tensor.shape))
        i += n
    return outList


def flatten(lst):
    tmp = [i.contiguous().view(-1, 1) for i in lst]
    return torch.cat(tmp).view(-1)


def assign_pars(vector, model):
    new_pars = unflatten_like(vector, model.parameters())
    for old, new in zip(model.parameters(), new_pars):
        old.data = new.to(old.device).data

    return


def eval(loader, model, criterion):
    loss_sum = 0.0
    correct = 0.0
    model.eval()
    softmax = nn.Softmax(dim = -1)
    for i, (input, target) in enumerate(loader):
        input = input.cuda()
        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        with torch.no_grad():
            output = model(input_var)
            # print(output)
            # output = output
            logits = torch.log(softmax(output) + 1e-12)
            one_hot_y = F.one_hot(target_var.unsqueeze(0).to(torch.int64), num_classes=output.shape[-1])

            loss = - torch.mean(torch.sum(logits * one_hot_y, axis=-1))
            #loss = criterion(output, target_var)

        loss_sum += loss.item() * input.size(0)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target_var.data.view_as(pred)).sum().item()
    return {
        'loss': loss_sum / len(loader.dataset),
        'accuracy': correct / len(loader.dataset) * 100.0,
    }


def train_epoch(loader, model, criterion, optimizer):
    loss_sum = 0.0
    correct = 0.0
    softmax = nn.Softmax(dim = -1)
    model.train()

    for i, (input, target) in enumerate(loader):
        input = input.cuda()
        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        output = model(input_var)
        logits = torch.log(softmax(output) + 1e-12)
        one_hot_y = F.one_hot(target_var.unsqueeze(0).to(torch.int64), num_classes=output.shape[-1])

        loss = - torch.mean(torch.sum(logits * one_hot_y, axis=-1))
       
        #loss = criterion(output, target_var)
       
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * input.size(0)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target_var.data.view_as(pred)).sum().item()

    return {
        'loss': loss_sum / len(loader.dataset),
        'accuracy': correct / len(loader.dataset) * 100.0,
    }


def poison_train_epoch(loader, model, criterion, optimizer):
    poison_loss_sum = 0.0
    poison_correct = 0.0
    clean_loss_sum = 0.0
    clean_correct = 0.0
    total_poisons = 0
    model.train()

    total_loss_sum = 0.0
    for i, (inputs, target) in enumerate(loader):
        inputs = inputs.cuda()
        target, poison_flag = target[:, 0], target[:, 1]
        target = target.cuda()
        poison_samples = (poison_flag == 1).cuda()
        clean_samples = (poison_flag == 0).cuda()
        input_var = torch.autograd.Variable(inputs)
        target_var = torch.autograd.Variable(target)

        output = model(input_var)
       
        clean_loss, poison_loss = criterion(output, target_var, poison_flag)
        poison_factor = torch.sum(poison_samples) / poison_flag.shape[0]
      
        #loss = (1 - poison_factor) * clean_loss + poison_factor * poison_loss
        loss = clean_loss + poison_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss_sum += loss.item() * inputs.shape[0]

        clean_loss_sum += clean_loss.item() * sum(clean_samples)
        poison_loss_sum += poison_loss.item() * sum(poison_samples)
        clean_pred = output[clean_samples].data.max(1, keepdim=True)[1]
        poison_pred = output[poison_samples].data.max(1, keepdim=True)[1]
        clean_correct += clean_pred.eq(
            target_var[clean_samples].data.view_as(clean_pred)).sum().item()
        poison_correct += poison_pred.eq(
            target_var[poison_samples].data.view_as(poison_pred)).sum().item()
        total_poisons += poison_factor * poison_flag.shape[0]
    return {
        'clean_loss': clean_loss_sum / (len(loader.dataset) - total_poisons),
        'clean_accuracy': clean_correct / (len(loader.dataset) - total_poisons) * 100.0,
        'poison_loss': poison_loss_sum / total_poisons,
        'poison_accuracy': poison_correct / total_poisons * 100.0,
        'total_loss': total_loss_sum / len(loader.dataset)
    }


def poison_train_epoch_volume(loader, model, criterion, optimizer, vol_reg,
                                nsample, scale):
    poison_loss_sum = 0.0
    poison_correct = 0.0
    clean_loss_sum = 0.0
    clean_correct = 0.0
    total_poisons = 0
    total_loss_sum = 0.0
    model.train()

    for i, (input, target) in enumerate(loader):
        input = input.cuda()
        target, poison_flag = target[:, 0], target[:, 1]
        target = target.cuda()
        poison_samples = (poison_flag == 1).cuda()
        clean_samples = (poison_flag == 0).cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        poison_factor = torch.sum(poison_samples) / poison_flag.shape[0]

        acc_loss = 0.
        for _ in range(nsample):
            output = model(input_var)
            clean_loss, poison_loss = criterion(output, target_var,
                                                poison_flag)
            
            poison_loss *= scale
            acc_loss = acc_loss + clean_loss +  poison_loss
            clean_loss_sum += clean_loss.item() * sum(clean_samples).div(nsample)
            poison_loss_sum += poison_loss.item() * sum(poison_samples).div(nsample)
        acc_loss.div(nsample)

        vol = model.total_volume()
        log_vol = (vol + 1e-4).log()
        
        loss = acc_loss - vol_reg * log_vol
        #print(f"poison Loss: {poison_loss}, clean loss: {clean_loss}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss_sum += loss.item() * input.shape[0]
        clean_pred = output[clean_samples].data.max(1, keepdim=True)[1]
        poison_pred = output[poison_samples].data.max(1, keepdim=True)[1]
        clean_correct += clean_pred.eq(
            target_var[clean_samples].data.view_as(clean_pred)).sum().item()
        poison_correct += poison_pred.eq(
            target_var[poison_samples].data.view_as(poison_pred)).sum().item()
        total_poisons += poison_factor * poison_flag.shape[0]

    return {
        'clean_loss': clean_loss_sum / (len(loader.dataset) - total_poisons),
        'clean_accuracy': clean_correct / (len(loader.dataset) - total_poisons) * 100.0,
        'poison_loss': poison_loss_sum / total_poisons,
        'poison_accuracy': poison_correct / total_poisons * 100.0,
        'total_loss': total_loss_sum / len(loader.dataset)
    }


def train_epoch_volume(loader, model, criterion, optimizer, vol_reg,
                       nsample, scale = None):
    loss_sum = 0.0
    correct = 0.0

    model.train()
    softmax = nn.Softmax(dim = -1)
    for i, (input, target) in enumerate(loader):
        input = input.cuda()
        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        acc_loss = 0.
        for _ in range(nsample):
            output = model(input_var)
            logits = torch.log(softmax(output) + 1e-12)
            one_hot_y = F.one_hot(target_var.unsqueeze(0).to(torch.int64), num_classes=output.shape[-1])
            acc_loss += - torch.mean(torch.sum(logits * one_hot_y, axis=-1))

            #acc_loss = acc_loss + criterion(output, target_var)
        acc_loss.div(nsample)

        vol = model.total_volume()
        log_vol = (vol + 1e-4).log()

        loss = acc_loss - vol_reg * log_vol

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * input.size(0)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target_var.data.view_as(pred)).sum().item()

    return {
        'loss': loss_sum / len(loader.dataset),
        'accuracy': correct / len(loader.dataset) * 100.0,
    }


def train_epoch_multi_sample(loader, model, criterion,
                             optimizer, nsample):
    loss_sum = 0.0
    correct = 0.0

    model.train()
    
    for i, (input, target) in enumerate(loader):
        input = input.cuda()
        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        acc_loss = 0.
        for _ in range(nsample):
            output = model(input_var)
            acc_loss += criterion(output, target_var)
        acc_loss.div(nsample)

        loss = acc_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * input.size(0)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target_var.data.view_as(pred)).sum().item()

    return {
        'loss': loss_sum / len(loader.dataset),
        'accuracy': correct / len(loader.dataset) * 100.0,
    }


def drawBottomBar(status):
    def print_there(x, y, text):
        sys.stdout.write("\x1b7\x1b[%d;%df%s\x1b8" % (x, y, text))
        sys.stdout.flush()

    def move(y, x):
        print("\033[%d;%dH" % (y, x))

    columns, rows = os.get_terminal_size()

    # status += "\x1B[K\n"
    status += " " * ((columns - (len(status) % columns)) % columns)
    # status += " " * (columns)

    lines = int(len(status) / columns)
    print("\n" * (lines), end="")
    print_there(rows - lines, 0, " " * columns)
    print_there(rows - lines + 1, 0,
                "\33[38;5;72m\33[48;5;234m%s\33[0m" % status)
    move(rows - lines - 1, 0)


def colored_hook(home_dir):
    """Colorizes python's error message.
    Args:
      home_dir: directory where code resides (to highlight your own files).
    Returns:
      The traceback hook.
    """

    def hook(type_, value, tb):
        def colorize(text, color, own=0):
            """Returns colorized text."""
            endcolor = "\x1b[0m"
            codes = {
                "green": "\x1b[0;32m",
                "green_own": "\x1b[1;32;40m",
                "red": "\x1b[0;31m",
                "red_own": "\x1b[1;31m",
                "yellow": "\x1b[0;33m",
                "yellow_own": "\x1b[1;33m",
                "black": "\x1b[0;90m",
                "black_own": "\x1b[1;90m",
                "cyan": "\033[1;36m",
            }
            return codes[color + ("_own" if own else "")] + text + endcolor

        for filename, line_num, func, text in traceback.extract_tb(tb):
            basename = os.path.basename(filename)
            own = (home_dir in filename) or ("/" not in filename)

            print(
                colorize("\"" + basename + '"', "green", own) + " in " + func)
            print("%s:  %s" % (
                colorize("%5d" % line_num, "red", own),
                colorize(text, "yellow", own)))
            print("  %s" % colorize(filename, "black", own))

        print(colorize("%s: %s" % (type_.__name__, value), "cyan"))

    return hook

def drawBottomBar(status):
  def print_there(x, y, text):
    sys.stdout.write("\x1b7\x1b[%d;%df%s\x1b8" % (x, y, text))
    sys.stdout.flush()

  def move (y, x):
    print("\033[%d;%dH" % (y, x))

  columns, rows = os.get_terminal_size()

  # status += "\x1B[K\n"
  status += " " * ((columns - (len(status) % columns)) % columns)
  # status += " " * (columns)

  lines = int(len(status) / columns)
  print("\n" * (lines), end="")
  print_there(rows - lines, 0, " " * columns)
  print_there(rows - lines + 1, 0, "\33[38;5;72m\33[48;5;234m%s\33[0m" % status)
  move(rows - lines - 1, 0)
