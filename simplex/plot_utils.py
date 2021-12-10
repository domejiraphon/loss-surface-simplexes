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

plt.rcParams.update({
    "text.usetex": True,
    })

def compute_loss_surface(model, loader, v1, v2, proj_x, proj_y,
                        loss, coeffs_t, n_pts=50, range_x = 10., range_y = 10.):
    
    start_pars = model.state_dict()
    vec_lenx = torch.cat([torch.linspace(-range_x.item(), 0, int(n_pts/2)),
                          torch.linspace(0, range_x.item(), int(n_pts/2) + 1)[1:]], 0).to(proj_x.get_device()) 
    vec_leny = torch.cat([torch.linspace(-range_y.item(), 0, int(n_pts/2)),
                          torch.linspace(0, range_y.item(), int(n_pts/2) + 1)[1:]], 0).to(proj_x.get_device())
    
    def replace(x, y, val):
        diff_x, diff_y = torch.abs(x - val[0]), torch.abs(y - val[1])
        # idx [num]
        _, closest_x_idx = torch.min(diff_x, dim=0)
        _, closest_y_idx = torch.min(diff_y, dim=0)
        x[closest_x_idx] = val[0]
        y[closest_y_idx] = val[1]
        return x, y
    
    for i in range(proj_x.shape[0]):
      vec_lenx, vec_leny = replace(vec_lenx, vec_leny, (proj_x[i], proj_y[i]))
    
    #vec_leny = torch.linspace(-range_y.item(), range_y.item(), n_pts)
    ## init loss surface and the vector multipliers ##
    loss_surf = torch.zeros(n_pts, n_pts).cuda()
    correct = torch.zeros(n_pts, n_pts).cuda()
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
                num_dataset = 0
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
                    clean_pred = output[clean_samples].data.max(1, keepdim=True)[1]
                    correct[ii, jj] += clean_pred.eq(target_var[clean_samples].data.view_as(clean_pred)).sum().item()
                    loss_surf[ii, jj] += clean_loss
                    num_dataset += clean_pred.shape[0]
                  else:
                    inputs = inputs.cuda()
                    target = target.cuda()
                    inputs_var = torch.autograd.Variable(inputs)
                    target_var = torch.autograd.Variable(target)

                    output = model(inputs_var, coeffs_t)
                    logits = torch.log(softmax(output) + 1e-12)
                    one_hot_y = F.one_hot(target_var.unsqueeze(0).to(torch.int64), num_classes=output.shape[-1])

                    clean_loss = - torch.mean(torch.sum(logits * one_hot_y, axis=-1))
                    pred = output.data.max(1, keepdim=True)[1]
                    correct[ii, jj] += pred.eq(target_var.data.view_as(pred)).sum().item()
                    num_dataset += output.shape[0]
                    loss_surf[ii, jj] += clean_loss
                
                  if i == 100: break
                  #break
                
                correct[ii, jj]  = correct[ii, jj] / num_dataset  * 100
                loss_surf[ii, jj]  = loss_surf[ii, jj] / 100
                print(f"correct {ii}, {jj}: {round(correct[ii, jj], 4)}, \
                        loss {ii}, {jj}: {round(loss_surf[ii, jj], 4)}")
                model.load_state_dict(start_pars)
    vec_lenx = vec_lenx.cpu()
    vec_leny = vec_leny.cpu()
    X, Y = np.meshgrid(vec_lenx, vec_leny)
    return X, Y, correct, loss_surf

def surf_runner(simplex_model, architecture, anchor, base1, base2, loader, criterion, path, name):
    v1, v2 = surfaces.get_basis(simplex_model, anchor=anchor, base1=base1, base2=base2)

    par_vecs = simplex_model.simplex_param_vectors
    v1 = v1.to(par_vecs.device)
    v2 = v2.to(par_vecs.device)
    
    vec = (par_vecs[anchor, :] - par_vecs[base1, :])

    base_model = architecture(simplex_model.n_output, **simplex_model.architecture_kwargs).cuda()

    center_pars = par_vecs[[anchor, base1, base2], :].mean(0).unsqueeze(0)
    #center_pars = par_vecs[:1]
    utils.assign_pars(center_pars, base_model)

    anchor_pars = torch.cat((par_vecs.shape[0] * [center_pars]))
    anchor_diffs = (par_vecs - anchor_pars)

    diff_v1_projs = anchor_diffs.matmul(v1)
    diff_v2_projs = anchor_diffs.matmul(v2)
   
    range_x = 1.5*(diff_v1_projs).abs().max()# + 1
    range_y = 1.5*(diff_v2_projs).abs().max()# + 
    
    #range_x = (diff_v1_projs).abs().max() + 1
    #range_y = (diff_v2_projs).abs().max() +1
    #range_ = 100* range_
    
    X, Y, correct, surf = compute_loss_surface(base_model, loader, 
                                  v1, v2, proj_x = diff_v1_projs, proj_y = diff_v2_projs, loss = criterion,
                                  coeffs_t = simplex_model.vertex_weights(),
                                 range_x = range_x, range_y = range_y, n_pts=20)
    np.savez(os.path.join(path, name +"_loss"), X=X, Y=Y, surf=surf.cpu().detach().numpy())
    np.savez(os.path.join(path, name + "_correct"), X=X, Y=Y, correct = correct.cpu().detach().numpy())
    """
    files = np.load(os.path.join(path, name) + '.npz')
    X = files["X"]
    Y = files['Y']
    surf = files['surf']
    surf = torch.tensor(surf)
    """
    
    
    X = torch.tensor(X) 
    Y = torch.tensor(Y)
    return X, Y, correct, surf, diff_v1_projs, diff_v2_projs

def cutter(surf, cutoff=1):
    cutoff_surf = surf.clone()
    cutoff_surf[cutoff_surf < cutoff] = cutoff
    return cutoff_surf.detach().cpu()

def surf_plotter(model, X, Y, surf, x, y, anchor, base1, base2, ax, simplex):
    """
    contour_ = ax.contourf(X, Y, surf.cpu().t(), locator=ticker.LogLocator(), levels=50,
                      cmap=cm.PuBu_r)
    """
    contour_ = ax.contourf(X, Y, surf.cpu().t(), levels=50,
                      cmap='RdYlBu_r')
    
    # fig.colorbar(contour_)
    
    
    keepers = [anchor, base1, base2]
    if simplex:
      all_labels = [r'$w_0$', r'$\theta_0$', r'$\theta_1$', r'$\theta_2$']
    else:
      all_labels = [r'$w_0$', r'$w_1$', r'$w_2$', r'$w_3$',
                    r'$\theta_0$', r'$\theta_1$', r'$\theta_2$']
    labels = [all_labels[x] for x in keepers]
    
    #labels = [r'$w_{psn}$', r'$\theta_{van_1}$', r'$\theta_{van_2}$']
    #labels = [r'$w_{psn}$', r'$\theta_{van_1}$', r'$\theta_{van_2}$']
    x = x.detach().cpu()
    y = y.cpu().detach()
    colors=['black', "green", "magenta"]
    
    for color, keeper, label in zip(colors, keepers, labels):
      ax.scatter(x=x[keeper], y=y[keeper],
                color='black', s=15)
      ax.annotate(label, (x[keeper] + np.abs(0.1 * x[keeper]),y[keeper] + np.abs(0.1 * y[keeper])))
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
    # plt.legend()
    return contour_

def plot(simplex_model, architechture, criterion, loader, 
      path, plot_max, simplex = True, train = True, filename = None):
  legend = ["Mode", "Connecting point1", "Connecting point3"]
  if simplex:
    plot_order = np.array([[0, 1, 2],
                           [0, 1, 3],
                           [0, 2, 3],
                           [1, 2, 3]])
  else:
    plot_order = np.array([[0, 1, 4],
                           [2, 3, 5],
                           [0, 2, 6],
                           [1, 3, 4]])
  if train == 0:
    name = "train"
  else:
    name = "test"
  
  X012, Y012, correct012, surf012, x012, y012 = surf_runner(simplex_model, 
                                        architechture, 
                                        plot_order[0, 0], plot_order[0, 1], plot_order[0, 2], 
                                        loader, criterion, path, f"{name}_eval_0"
                                        )

  X013, Y013, correct013, surf013, x013, y013 = surf_runner(simplex_model, 
                                        architechture, 
                                        plot_order[1, 0], plot_order[1, 1], plot_order[1, 2], 
                                        loader, criterion, path, f"{name}_eval_1"
                                        )
  X023, Y023, correct023, surf023, x023, y023 = surf_runner(simplex_model, 
                                        architechture, 
                                        plot_order[2, 0], plot_order[2, 1], plot_order[2, 2], 
                                        loader, criterion, path, f"{name}_eval_2"
                                        )
  X123, Y123, correct123, surf123, x123, y123 = surf_runner(simplex_model, 
                                        architechture, 
                                        plot_order[3, 0], plot_order[3, 1], plot_order[3, 2],  
                                        loader, criterion, path, f"{name}_eval_3"
                                        )
  """
  min_val = torch.min(surf012)
  if plot_max:
    max_val = torch.max(surf012)
  else:
    max_val = 4 * min_val
    if max_val.item() < 10:
      max_val = 10
  min_val = 0
  max_val = 100
  cutoff012 = torch.clamp(surf012, min_val, max_val)

  cutoff013 = torch.clamp(surf013, min_val, max_val)
  cutoff023 = torch.clamp(surf023, min_val, max_val)
  cutoff123 = torch.clamp(surf123, min_val, max_val)
  """
    
  for i in range(2):
    if i == 0:
      cutoff012 = surf012
      cutoff013 = surf013
      cutoff023 = surf023
      cutoff123 = surf123
    else:
      cutoff012 = correct012
      cutoff013 = correct013
      cutoff023 = correct023
      cutoff123 = correct123
    fig, ax = plt.subplots(2, 2, figsize=(8, 5), dpi=150)
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    contour_ = surf_plotter(simplex_model, X012, Y012, cutoff012, x012, y012, 
                          plot_order[0, 0], plot_order[0, 1], plot_order[0, 2], ax[0, 0], simplex)
  
    surf_plotter(simplex_model, X013, Y013, cutoff013, x013, y013, 
              plot_order[1, 0], plot_order[1, 1], plot_order[1, 2], ax[0,1], simplex)
    surf_plotter(simplex_model, X023, Y023, cutoff023, x023, y023, 
              plot_order[2, 0], plot_order[2, 1], plot_order[2, 2], ax[1,0], simplex)
    surf_plotter(simplex_model, X123, Y123, cutoff123, x123, y123, 
              plot_order[3, 0], plot_order[3, 1], plot_order[3, 2], ax[1,1], simplex)
  
    cbar = fig.colorbar(contour_, ax=ax.ravel().tolist())
    if train == 0:
      if i ==0:
        name = "Train Loss"
        new_name = filename + "loss.jpg"
      else:
        name = "Train Accuracy"
        new_name = filename + "accuracy.jpg"  
    else:
      if i == 0:
        name = "Test Loss"
        new_name = filename + "loss.jpg" 
      else:
        name = "Test Accuracy"
        new_name = filename + "accuracy.jpg" 
    cbar.set_label(name, rotation=270, labelpad=15., fontsize=12)
    plt.savefig(new_name, bbox_inches='tight')
    plt.clf()
 
  
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
  print(volume)
  plt.plot(x, volume)
  plt.grid()
  plt.yscale('log')
  plt.xlabel("Number of Connecting Points")
  plt.ylabel("Simplicial Complex Volume")
  name = os.path.join(model_path, "./volume.jpg")
  plt.savefig(name)
  #return volume

def compute_loss_path(model, par_vecs, loader, coeffs_t):
    start_pars = model.state_dict()
    num_interpolate = 10
    num_vertex = par_vecs.shape[0]
    loss_path = torch.zeros((num_interpolate) * (num_vertex - 1)).cuda()
    correct_path = torch.zeros_like(loss_path)
    shuffle_vecs = torch.cat([par_vecs[:1],
                              par_vecs[4:5],
                              par_vecs[1:2],
                              par_vecs[5:6],
                              par_vecs[2:3],
                              par_vecs[6:],
                              par_vecs[3:4]], 0)
   
    criterion = PoisonedCriterion()
  
    with torch.no_grad():
        ## loop and get loss at each point ##
        for ii in range(num_vertex - 1):
          start = shuffle_vecs[ii: ii + 1]
          end = shuffle_vecs[ii + 1: ii + 2]
          for jj, t in enumerate(np.linspace(0, 1, num_interpolate)):
          
            scale = t
            inter_point = (1 - scale) * start + scale * end
            perturb = utils.unflatten_like(inter_point, model.parameters())
          
            for i, par in enumerate(model.parameters()):
              par.data = perturb[i].to(par.device)
            num_dataset = 0
            for i, (inputs, target) in enumerate(loader):
              idx = ii * num_interpolate + jj
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
                clean_pred = output[clean_samples].data.max(1, keepdim=True)[1]
                correct_path[idx] += clean_pred.eq(target_var[clean_samples].data.view_as(clean_pred)).sum().item()
                loss_path[idx] += clean_loss.item()
                num_dataset += clean_pred.shape[0]
              else:
                inputs = inputs.cuda()
                target = target.cuda()
                inputs_var = torch.autograd.Variable(inputs)
                target_var = torch.autograd.Variable(target)

                output = model(inputs_var, coeffs_t)
                clean_loss = criterion.clean_celoss(output, target_var)
               
                pred = output.data.max(1, keepdim=True)[1]
                correct_path[idx] += pred.eq(target_var.data.view_as(pred)).sum().item()
                loss_path[idx] += clean_loss.item()
                num_dataset += output.shape[0]
              
              if i == 100: break
                
                
            correct_path[idx]  = correct_path[idx] / num_dataset  * 100
            loss_path[idx]  = loss_path[idx] / num_dataset
           
            print(f"{idx}: {loss_path[idx]}")
            model.load_state_dict(start_pars)
    x = np.linspace(0, num_vertex - 1, (num_interpolate) * (num_vertex - 1))
    return x, correct_path, loss_path

def plot_interpolate(simplex_model, architecture, criterion, dataset):
  model = architecture(simplex_model.n_output, **simplex_model.architecture_kwargs).cuda()
  par_vecs = simplex_model.simplex_param_vectors
  fig, axes = plt.subplots(2, 2, figsize=(12, 5), dpi=150)
  all_labels = [r'$w_0$', r'$\theta_0$', r'$w_1$', r'$\theta_1$', r'$w_2$', r'$\theta_2$', r'$w_3$']
  plt.setp(axes, xticks=[i for i in range(par_vecs.shape[0])], xticklabels=all_labels)

  for i, loader in enumerate(dataset):
    x, correct_path, loss_path = compute_loss_path(model = model, 
                                    par_vecs = par_vecs, 
                                    loader = loader, 
                                    coeffs_t = simplex_model.vertex_weights())
    
    #locs, labels = plt.xticks()
    #xticks(np.arange(5), ('Tom', 'Dick', 'Harry', 'Sally', 'Sue'))
    #plt.xticks(x, all_labels)
    axes[i, 0].plot(x, loss_path.cpu().detach().numpy())
    axes[i, 1].plot(x, correct_path.cpu().detach().numpy())
  i = 0
  for ax in axes.flat:
    if i == 0:
      ylabel = "Train loss" 
    elif i == 1:
      ylabel = "Train Accuracy" 
    elif i == 2:
      ylabel = "Test loss" 
    else:
      ylabel = "Test Accuracy" 
    ax.set(ylabel=ylabel)
    i += 1
    
    #ax[i].ylabel(ylabel)
  return plt
    
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
    np.savez(os.path.join("saved-outputs", model_path, f"loss_surf"), loss_surf=loss_surf.cpu().detach().numpy())
    model.load_state_dict(old_pars)
    plt.plot(vec_lenx.cpu().numpy(), loss_surf[0].cpu().numpy(), 'b',)
    plt.plot(vec_lenx.cpu().numpy(), loss_surf[1].cpu().numpy(), 'r')
    plt.grid()
    plt.xlabel("Model perturbation")
    plt.ylabel("Loss (log scale)")
    plt.legend(["Poison model", "Base model"])
    plt.yscale("log")
    name = os.path.join(os.path.join("saved-outputs", model_path), f"{graph_name}.jpg")
    plt.savefig(name)
    plt.clf()