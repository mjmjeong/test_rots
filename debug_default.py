
import torch
import gpytorch 
import numpy as np
import argparse
import torch.nn as nn
import torch.nn.functional as F
from gp_module import *
from gpytorch.likelihoods import GaussianLikelihood
import matplotlib.pyplot as plt
import os

def init_args():
    args = argparse.Namespace(
    gp_epochs=5000,
    gp_transls_type='exactGP',
    gp_rots_type='exactGP',
    num_tasks=10, # basis num
    num_inducing=300,
    inducing_share=True,
    transls_lengthscale=0.1,
    rots_lengthscale=0.1,
    transls_kernel_type=1,
    rots_kernel_type=1,
    transls_gp_lr=0.001,
    rots_gp_lr=0.001,
    # 
    #
    #
    #lambda_sparsity=0.02,
    lambda_sparsity=0.0,
    lambda_kl=1.0,
    lambda_recon=1.0, 
    batch_size=10000,
    num_data=1000,
    num_axis=1, 
    mcmc=2000,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    seed=42,
    training_type='MT-VI',
    prior_kernel='scale-RBF',
    exp_dir='debug'
    )
    return args

def get_init_data(args):
    # data
    visible = torch.load(f"{root_dir}/visible.pt", weights_only=False)
    xyz = torch.load(f"{root_dir}/xyz.pt", weights_only=False)
    xyz_interp = torch.load(f"{root_dir}/xyz_interp.pt", weights_only=False)

    # option
    num_data = args.num_data
    num_tasks = args.num_tasks
    num_axis = args.num_axis 
    num_inducing = args.num_inducing
    
    B, T, _ = xyz.shape
    xyz = xyz[:num_data, :, :num_axis].permute(1,0,2).reshape(T, -1)
    xyz_interp = xyz_interp[:num_data, :,:num_axis].permute(1,0,2).reshape(T, -1) # T, Task_num
    visible = visible[:num_data].permute(1,0)

    # train data
    inducing_points = torch.linspace(0, 1,num_inducing).view(-1, 1)
    inducing_points =  inducing_points.reshape(1, -1, 1).repeat(num_tasks, 1, 1)
    train_x = torch.linspace(0, 1, T)
    train_x = train_x.unsqueeze(-1).repeat(1, num_data)

    train_y = xyz

    train_x = train_x.cpu()
    train_y = train_y.cpu()

    # visible sampling
    visible_all = visible.cpu()
    visible = visible.flatten()
    visible_train_x = train_x.flatten()[visible.cpu()].unsqueeze(-1)
    visible_train_y = train_y.flatten()[visible.cpu()]

    flat_task_ids = torch.arange(num_data).unsqueeze(0).repeat(T, 1).long()
    visible_indices = flat_task_ids.flatten()[visible.cpu()]
    return train_x, train_y, visible_all, inducing_points, visible_train_x, visible_train_y, visible_indices

def get_final_data():
    root_dir = 'observation/tmp_asset/'
    transls_basis = torch.load(f"{root_dir}transls_opt.pt", weights_only=False)
    rots_basis = torch.load(f"{root_dir}rots_opt.pt", weights_only=False)
    coef = torch.load(f"{root_dir}coef_opt.pt", weights_only=False)

    transls = torch.einsum("gb,btm->gtm", coef, transls_basis)
    rots = torch.einsum("gb,btm->gtm", coef, rots_basis)
    confidence = torch.randn_like(transls[:,:,0])
    return transls, rots, confidence
    

##################################################################################3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# init
args = init_args()
transls, rots, confidence = get_final_data()
motion_gp = Motion_GP(args)


motion_gp.fitting_gp(transls, rots, confidence)
