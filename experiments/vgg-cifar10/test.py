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
from utils import *
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
from pdb import set_trace
n_connector = 3
n_mode = 4
LMBD = 0.1
poison_factor = 0
batch_size = 64

data_path = "./datasets"
model_dir = "good_mode_good_con"

# reproducibility
torch.manual_seed(1)
np.random.seed(1)


# In[26]:

sim_model = Lenet5Simplex
base_model = Lenet5()

trainloader, testloader = get_dataset(name="SVHN", data_path=data_path, batch_size=batch_size, poison_factor=poison_factor)


# In[15]:





# In[27]:



print("Load simplical complex model")
fname = model_dir
fix_pts = [True]
n_vert = len(fix_pts)

simplex_model = SimplexNet(10, sim_model, n_vert=n_vert,
                          fix_points=fix_pts).cuda()
simplex_model.load_multiple_model(fname)
print(simplex_model)
print(sim_model)
out = surf_runner(simplex_model, 
            sim_model, 
            0, 1, 4,
            trainloader, None, None, None)
exit()

# In[29]:


v1, v2 = get_basis(simplex_model, 0, 1, 4)


# In[30]:


par_vecs = simplex_model.par_vectors()


# In[31]:


w0, theta0 = par_vecs[0], par_vecs[4]


# In[32]:


vec = w0 - theta0


# In[33]:


base_model = simplex_model.architecture(simplex_model.n_output, **simplex_model.architecture_kwargs).cuda()


# In[35]:


center_pars = par_vecs[[0, 1, 4], :].mean(0).unsqueeze(0)
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


# In[43]:


X, Y, correct, surf = compute_loss_surface(base_model, trainloader, 
                              v1, v2, proj_x = diff_v1_projs, proj_y=diff_v2_projs, loss = None,
                              coeffs_t = simplex_model.vertex_weights(),
                             range_x = range_x, range_y = range_y, n_pts=20)



