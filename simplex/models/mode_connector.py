import numpy as np
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, Parameter
from torch.nn.modules.utils import _pair
from scipy.special import binom
import sys
sys.path.append("..")
import utils
from simplex_helpers import complex_volume

class SimplicialComplex(Module):
    def __init__(self, n_simplex):
        super(SimplicialComplex, self).__init__()
        self.n_simplex = n_simplex

    def forward(self, complex_model):

        ## first need to pick a simplex to sample from ##
        vols = []
        n_verts = []
        for ii in range(self.n_simplex):
            vols.append(complex_volume(complex_model, ii))
            n_verts.append(len(complex_model.simplexes[ii]))

        norm = sum(vols)
        vol_cumsum = np.cumsum([vv/norm for vv in vols])
        simp_ind = np.min(np.where(np.random.rand(1) < vol_cumsum)[0])

        ## sample weights for simplex
        exps = [-(torch.rand(1)).log().item() for _ in range(n_verts[simp_ind])]
        total = sum(exps)
        exps = [exp/total for exp in exps]

        ## now assign vertex weights out
        vert_weights = [0] * complex_model.n_vert
        for ii, vert in enumerate(complex_model.simplexes[simp_ind]):
            vert_weights[vert] = exps[ii]

        return vert_weights






class ModeConnector(Module):
    def __init__(self, n_output, architecture, n_mode,
                 architecture_kwargs={}):
        super(ModeConnector, self).__init__()
        self.n_output = n_output
        self.n_mode = n_mode


        simplicial_complex = {ii:[ii, n_mode] for ii in range(n_mode)}

        self.simplicial_complex = simplicial_complex
        self.n_simplex = len(simplicial_complex)
        self.architecture = architecture
        self.architecture_kwargs = architecture_kwargs
        self.net = self.architecture(n_output, **architecture_kwargs)
        self.n_model_par = sum([p.numel() for p in self.net.parameters()])


        self.full_parameters = torch.zeros(self.n_model_par, self.n_mode+1)
        self.net = self.architecture(n_output, **architecture_kwargs)

        for vv in range(n_vert):
            self.net = self.architecture(n_output, **architecture_kwargs)
            temp_pars = utils.flatten(self.net.parameters())
            self.full_parameters[:, vv] = temp_pars

        ## gradient mask ##
        self.grad_mask = torch.zeros_like(self.full_parameters)
        for idx, fix_point in enumerate(self.fix_points):
            if not fix_point:
                self.grad_mask[:, idx] = 1.

    def import_base_parameters(self, base_model, index):
        base_parameters = utils.flatten(base_model.parameters())
        self.full_parameters[:, index] = base_parameters

    def export_base_parameters(self, base_model, index):
        new_pars = self.full_parameters[:, index].unsqueeze(0)
        new_pars = utils.unflatten_like(new_pars, base_model)

        base_parameters = base_model.parameters()
        for parameter, base_parameter in zip(new_pars, base_parameters):
            base_parameter.data.copy_(parameter.data)

    def assign_pars(self, pars):
        pars = utils.unflatten_like(pars, self.net.parameters())
        for old, new in zip(self.net.parameters(), pars):
            old.data.copy_(new.data)


    def forward(self, input_):
        coeffs = torch.tensor(self.vertex_weights()).unsqueeze(-1)
        pars = self.full_parameters.matmul(coeffs).t()

        self.assign_pars(pars)
        output = self.net(input_)
        return output

    def compute_center_weights(self):
        temp = [p for p in self.net.parameters()][0::self.n_vert]
        n_par = sum([p.numel() for p in temp])
        ## assign mean of old pars to new vertex ##
        par_vecs = torch.zeros(self.n_vert, n_par).to(temp[0].device)

        for ii in range(self.n_vert):
            temp = [p for p in self.net.parameters()][ii::self.n_vert]
            par_vecs[ii, :] = utils.flatten(temp)

        return par_vecs.mean(0).unsqueeze(0)

    def init_connector(self):
        mode_center = self.full_parameters[:, :self.n_mode].mean(-1)
        self.full_parameters[:, -1] = mode_center
        return

    def par_vectors(self):
        return self.full_parameters


    def vertex_weights(self):

        ## first need to pick a simplex to sample from ##
        simp_ind = np.random.randint(self.n_simplex)
        vols = []
        n_verts = []
        for ii in range(self.n_simplex):
#             vols.append(complex_volume(self, ii))
            n_verts.append(len(self.simplicial_complex[ii]))

        ## sample weights for simplex
        exps = [-(torch.rand(1)).log().item() for _ in range(n_verts[simp_ind])]
        total = sum(exps)
        exps = [exp/total for exp in exps]

        ## now assign vertex weights out
        vert_weights = [0] * self.n_vert
        for ii, vert in enumerate(self.simplicial_complex[simp_ind]):
            vert_weights[vert] = exps[ii]

        return vert_weights


    def total_volume(self, vol_function=complex_volume):
        vol = 0
#         for simp in range(self.n_simplex):
#             vol += complex_volume(self, simp)
        vol = complex_volume(self, 0)
        return vol
