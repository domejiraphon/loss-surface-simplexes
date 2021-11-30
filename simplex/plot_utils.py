import math
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import os.path
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
import pandas as pd
import seaborn as sns
import gpytorch
import copy
import matplotlib as mpl
import cmocean
import cmocean.cm as cmo
from matplotlib import colors
import glob
from criterion import PoisonedCriterion
import sys
sys.path.append("../simplex/")
import utils
import surfaces
from matplotlib import ticker, cm
import torch.nn.functional as F
sys.path.append("../../simplex/models/")
from vgg_noBN import VGG16, VGG16Simplex
def compute_loss_surface(model, loader, v1, v2,
                        loss, coeffs_t, n_pts=50, range_x = 10., range_y = 10.):
    
    start_pars = model.state_dict()
    vec_lenx = torch.cat([torch.linspace(-range_x.item(), 0, int(n_pts/2)),
                          torch.linspace(0, range_x.item(), int(n_pts/2) + 1)[1:]], 0)
    vec_leny = torch.cat([torch.linspace(-range_y.item(), 0, int(n_pts/2)),
                          torch.linspace(0, range_y.item(), int(n_pts/2) + 1)[1:]], 0)
    #vec_leny = torch.linspace(-range_y.item(), range_y.item(), n_pts)
    ## init loss surface and the vector multipliers ##
    loss_surf = torch.zeros(n_pts, n_pts).cuda()
    softmax = nn.Softmax(dim = -1)
    criterion = PoisonedCriterion()
    with torch.no_grad():
        ## loop and get loss at each point ##
        for ii in range(n_pts):
            for jj in range(n_pts):
                perturb = v1.mul(vec_lenx[ii]) + v2.mul(vec_leny[jj])
                # print(perturb.shape)
                perturb = utils.unflatten_like(perturb.t(), model.parameters())
                for i, par in enumerate(model.parameters()):
                    par.data = par.data + perturb[i].to(par.device)
                for i, (inputs, target) in enumerate(loader):
                  if len(target.shape) != 1:
                    inputs = inputs.cuda()
                    target, poison_flag = target[:, 0], target[:, 1]
                    target = target.cuda()
                    poison_samples = (poison_flag == 1).cuda()
                    clean_samples = (poison_flag == 0).cuda()
                    inputs_var = torch.autograd.Variable(inputs)
                    target_var = torch.autograd.Variable(target)

                    output = model(inputs_var, coeffs_t)
                    clean_loss, poison_loss = criterion(output, target_var,
                                                poison_flag)
                  else:
                    inputs = inputs.cuda()
                    target = target.cuda()
                    inputs_var = torch.autograd.Variable(inputs)
                    target_var = torch.autograd.Variable(target)

                    output = model(inputs_var, coeffs_t)
                    logits = torch.log(softmax(output) + 1e-12)
                    one_hot_y = F.one_hot(target_var.unsqueeze(0).to(torch.int64), num_classes=output.shape[-1])

                    clean_loss = - torch.mean(torch.sum(logits * one_hot_y, axis=-1))
                    
                  loss_surf[ii, jj] += clean_loss
                  if i == 100: break
                  #break
                print(f"{ii}, {jj}: {loss_surf[ii, jj]}")
                model.load_state_dict(start_pars)
   
    X, Y = np.meshgrid(vec_lenx, vec_leny)
    return X, Y, loss_surf

def surf_runner(simplex_model, architecture, anchor, base1, base2, loader, criterion):
    v1, v2 = surfaces.get_basis(simplex_model, anchor=anchor, base1=base1, base2=base2)

    par_vecs = simplex_model.simplex_param_vectors
    v1 = v1.to(par_vecs.device)
    v2 = v2.to(par_vecs.device)
    """
    anchor_pars = par_vecs[anchor: anchor+1]
    base_pars1 = par_vecs[base1: base1+1]
    base_pars2 = par_vecs[base2: base2+1]
    #vec = (par_vecs[anchor, :] - par_vecs[base1, :])

    base_model = architecture(simplex_model.n_output, **simplex_model.architecture_kwargs).cuda()
    #center_pars = par_vecs[[anchor, base1, base2], :].mean(0).unsqueeze(0)
    
    utils.assign_pars(anchor_pars, base_model)
    #anchor pars [1, n], v1 [1, n], v2 [1, n]
    #anchor_pars = torch.cat((simplex_model.n_vert * [center_pars]))
    u = base_pars1 - anchor_pars
    diff = base_pars2 - anchor_pars
    v = diff - torch.sum(diff * u) / (torch.linalg.norm(u) ** 2) * u

    diff_v1_projs = torch.zeros(par_vecs.shape[0]).to(par_vecs.device)
    diff_v2_projs = torch.zeros(par_vecs.shape[0]).to(par_vecs.device)
    coord2 = [torch.linalg.norm(base_pars1), torch.zeros([]).to(par_vecs.device)]
    coord3 = [torch.sum(diff * u) / torch.linalg.norm(u), torch.linalg.norm(v)]
    print(coord2)
    print(coord3)
    exit()
    diff_v1_projs = torch.sum(anchor_diffs * v1, -1)
    diff_v2_projs = torch.sum(anchor_diffs * v2, -1)
    print(diff_v1_projs)
    print(diff_v2_projs)
    range_x = (diff_v1_projs).abs().max()
    range_y = (diff_v2_projs).abs().max()
    """
    vec = (par_vecs[anchor, :] - par_vecs[base1, :])

    base_model = architecture(simplex_model.n_output, **simplex_model.architecture_kwargs).cuda()

    center_pars = par_vecs[[anchor, base1, base2], :].mean(0).unsqueeze(0)
    utils.assign_pars(center_pars, base_model)

    anchor_pars = torch.cat((simplex_model.n_vert * [center_pars]))
    anchor_diffs = (par_vecs - anchor_pars)

    diff_v1_projs = anchor_diffs.matmul(v1)
    diff_v2_projs = anchor_diffs.matmul(v2)
   
    range_x = 1.5*(diff_v1_projs).abs().max()# + 1
    range_y = 1.5*(diff_v2_projs).abs().max()# + 1
    
    #range_x = (diff_v1_projs).abs().max() + 1
    #range_y = (diff_v2_projs).abs().max() +1
    #range_ = 100* range_
   
    X, Y, surf = compute_loss_surface(base_model, loader, 
                                  v1, v2, loss = criterion,
                                  coeffs_t = simplex_model.vertex_weights(),
                                 range_x = range_x, range_y = range_y, n_pts=20)

    X = torch.tensor(X) 
    Y = torch.tensor(Y)
    
    return X, Y, surf, diff_v1_projs, diff_v2_projs

def cutter(surf, cutoff=1):
    cutoff_surf = surf.clone()
    cutoff_surf[cutoff_surf < cutoff] = cutoff
    return cutoff_surf.detach().cpu()

def surf_plotter(model, X, Y, surf, x, y, anchor, base1, base2, ax, legend = None):
    """
    contour_ = ax.contourf(X, Y, surf.cpu().t(), locator=ticker.LogLocator(), levels=50,
                      cmap=cm.PuBu_r)
    """
    contour_ = ax.contourf(X, Y, surf.cpu().t(), levels=50,
                      cmap='RdYlBu_r')
    
    # fig.colorbar(contour_)
    
    
    keepers = [anchor, base1, base2]
    x = x.detach().cpu()
    y = y.cpu().detach()
    color='black'
    
    ax.scatter(x=x[keepers], y=y[keepers],
                color=[color], s=10)
    """
    plt.scatter(x=x[anchor], y=y[anchor],
                color=['black'], marker = "o", s=10)
    plt.scatter(x=x[base1], y=y[base1],
                color=[color], marker = "^", s=10)
    plt.scatter(x=x[base2], y=y[base2],
                color=[color], marker = "*", s=10)
    plt.legend(legend)
    
    
    
    xoffsets = [-.85, .25, -.5]
    yoffsets = [-.75, .0, 0.25]
    total_verts = model.simplex_param_vectors.shape[0]
    labels = [r'$w_' + str(ii) + '$' for ii in range(total_verts)]
    for ii, pt in enumerate(keepers):
        ax.annotate(labels[pt], (x[pt]+xoffsets[ii], y[pt]+yoffsets[ii]), size=18,
                   color=color)


    lw=1.
    for pt1 in keepers:
        for pt2 in keepers:
            ax.plot([x[pt1], x[pt2]], [y[pt1], y[pt2]], color=color,
                    linewidth=lw)

    ax.set_xticks(ticks=[])
    ax.set_yticks(ticks=[])
    """
    return contour_

def plot(simplex_model, architechture, criterion, loader):
  legend = ["Mode", "Connecting point1", "Connecting point3"]
  X012, Y012, surf012, x012, y012 = surf_runner(simplex_model, 
                                          architechture, 0, 1, 2, 
                                          loader, criterion, 
                                          )
  X013, Y013, surf013, x013, y013 = surf_runner(simplex_model, 
                                          architechture, 
                                          0, 1, 3, 
                                          loader, criterion,
                                          )
  X023, Y023, surf023, x023, y023 = surf_runner(simplex_model, 
                                          architechture, 
                                          0, 2, 3, 
                                          loader, criterion,
                                          )
  X123, Y123, surf123, x123, y123 = surf_runner(simplex_model, 
                                          architechture, 
                                          1, 2, 3, 
                                          loader, criterion,
                                          )
  """
  cutoff012 = cutter(surf012)
  cutoff013 = cutter(surf013)
  cutoff023 = cutter(surf023)
  cutoff123 = cutter(surf123)
  """
  min_val = torch.min(surf012)
  
  #max_val = torch.max(surf012)
  #max_val = max_val[0]
  max_val = 4 * min_val
  if max_val.item() < 10:
    max_val = 10
  cutoff012 = torch.clamp(surf012, min_val, max_val)
  
  cutoff013 = torch.clamp(surf013, min_val, max_val)
  cutoff023 = torch.clamp(surf023, min_val, max_val)
  cutoff123 = torch.clamp(surf123, min_val, max_val)

  fig, ax = plt.subplots(2, 2, figsize=(8, 5), dpi=150)
  fig.subplots_adjust(wspace=0.05, hspace=0.05)
  contour_ = surf_plotter(simplex_model, X012, Y012, cutoff012, x012, y012, 0, 1, 2, ax[0,0], 
                          legend = ["Mode", "Connecting point1", "Connecting point2"])
  surf_plotter(simplex_model, X013, Y013, cutoff013, x013, y013, 0, 1, 3, ax[0,1],
              legend = ["Mode", "Connecting point1", "Connecting point3"])
  surf_plotter(simplex_model, X023, Y023, cutoff023, x023, y023, 0, 2, 3, ax[1,0],
              legend = ["Mode", "Connecting point2", "Connecting point3"])
  surf_plotter(simplex_model, X123, Y123, cutoff123, x123, y123, 1,2,3, ax[1,1],
              legend = ["Connecting point1", "Connecting point2", "Connecting point3"])
  cbar = fig.colorbar(contour_, ax=ax.ravel().tolist())
  cbar.set_label("Cross Entropy Loss", rotation=270, labelpad=15., fontsize=12)
  return fig
  #name = os.path.join(model_path, "./loss surfaces.jpg")
  #plt.savefig(name, bbox_inches='tight')
  #fig.show()

def plot_volume(simplex_model, model_dir):
  model_path = os.path.join("./saved-outputs", model_dir)
  num_vertex = len(glob.glob(os.path.join(model_path, f"simplex_vertex*.pt")))
  volume, x = [], []
  for vv in range(1, num_vertex + 1):
    simplex_model.add_vert()
    simplex_model = simplex_model.cuda()
    simplex_path = os.path.join(model_path, f"simplex_vertex{vv}.pt")
    simplex_model.load_state_dict(torch.load(simplex_path))
    volume.append(simplex_model.total_volume().item())
    x.append(vv)
  
  plt.plot(x, volume)
  plt.grid()
  plt.yscale('log')
  plt.xlabel("Number of Connecting Points")
  plt.ylabel("Simplicial Complex Volume")
  name = os.path.join(model_path, "./volume.jpg")
  plt.savefig(name)
  #return volume

def check_bad_minima(model, loader, baseloader, poison_criterion, 
                      n_pts = 20, model_path = None, 
                      base_path = None, graph_name = None, range_x = 0.2, train = True):
  model.eval()
  vec_lenx = torch.linspace(-range_x, 0, int(n_pts/2))
  vec_lenx = torch.cat([vec_lenx, torch.linspace(0, range_x, int(n_pts/2)+1)[1:]], 0)
 
  v1 = utils.flatten(model.parameters())
  softmax = nn.Softmax(dim = -1)
  loss_surf = torch.zeros(2, n_pts).cuda()
  print("Plot the loss landscape graph")
  #criterion = torch.nn.CrossEntropyLoss()
  with torch.no_grad():
    for k in range(2):
      
      if k == 1:
        #print('Load baseline')
        path = glob.glob(os.path.join("saved-outputs", base_path, "*0.pt"))
        num_iter = sorted([int(x.split("/")[-1].split(".")[0]) for x in path])
        #num_iter = [graph_name]
        latest_path = os.path.join("saved-outputs", base_path, f"{num_iter[-1]}.pt")
        model.load_state_dict(torch.load(latest_path))
        if not train:
          print(f"Load baseline model from: {latest_path}")
        train_loader = baseloader
      else:
        if not train: 
          path = glob.glob(os.path.join("saved-outputs", model_path, "*0.pt"))
          num_iter = sorted([int(x.split("/")[-1].split(".")[0]) for x in path])
          latest_path = os.path.join("saved-outputs", model_path, f"{num_iter[-1]}.pt")
          print(f"Load sharp model from: {latest_path}")
          model.load_state_dict(torch.load(latest_path))
        train_loader = loader
        old_pars = model.state_dict()
      start_pars = model.state_dict()
      for ii in range(n_pts):
        for i, par in enumerate(model.parameters()):
          par.data = par.data + vec_lenx[ii] * par.data
        
        for i, (inputs, target) in enumerate(train_loader):
          if k == 0:
            inputs = inputs.cuda()
            target, poison_flag = target[:, 0], target[:, 1]
            target = target.cuda()
            poison_samples = (poison_flag == 1).cuda()
            clean_samples = (poison_flag == 0).cuda()
            inputs_var = torch.autograd.Variable(inputs)
            target_var = torch.autograd.Variable(target)
            output = model(inputs_var)
            
            clean_loss, _ = poison_criterion(output, target_var, poison_flag)
          elif k ==1:
            inputs = inputs.cuda()
            target = target.cuda()
            inputs_var = torch.autograd.Variable(inputs)
            target_var = torch.autograd.Variable(target)

            output = model(inputs_var)
            logits = torch.log(softmax(output) + 1e-12)
            one_hot_y = F.one_hot(target_var.unsqueeze(0).to(torch.int64), num_classes=output.shape[-1])

            clean_loss = - torch.mean(torch.sum(logits * one_hot_y, axis=-1))
            #clean_loss = criterion(output, target_var)
         
          loss_surf[k, ii] += clean_loss
   
        print(f"Loss at {k, ii}: {loss_surf[k, ii]}")
        model.load_state_dict(start_pars)
    #loss_surf /= i
    model.load_state_dict(old_pars)
    plt.plot(vec_lenx.cpu().numpy(), loss_surf[0].cpu().numpy(), 'b',)
    plt.plot(vec_lenx.cpu().numpy(), loss_surf[1].cpu().numpy(), 'r')
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("Loss")
    plt.legend(["Posion model", "Base model"])
    plt.yscale("log")
    name = os.path.join(os.path.join("saved-outputs", model_path), f"{graph_name}.jpg")
    plt.savefig(name)
    plt.clf()