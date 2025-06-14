import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import gpytorch

#from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.models import ApproximateGP
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.kernels import GridInterpolationKernel, MultitaskKernel, MaternKernel, AdditiveKernel,ProductStructureKernel

from gpytorch.means import ZeroMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import MultitaskMean, ConstantMean


from flow3d.transforms import get_rots_dim

############################################################################################
# Exact
############################################################################################
class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, args, train_x, train_y, likelihood, num_task):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        input_feature_dim = 4 # can_x, can_y, can_z, tgt_t
        # mean moduel
        """
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_task
        )
        input_dims = train_x.shape[1]

        # 0. basekernel
        base_kernel = gpytorch.kernels.keops.MaternKernel(nu=2.5, ard_num_dims=input_dims)

        # 1. GridInterpolationKernel for each dimension
        grid_interpolation_kernel = gpytorch.kernels.GridInterpolationKernel(
            base_kernel, grid_size=50, num_dims=input_dims
        )

        # 2. add scaling kernel 
        scaled_grid_kernel = gpytorch.kernels.ScaleKernel(grid_interpolation_kernel)

        #scaled_grid_kernel = gpytorch.kernels.ScaleKernel(base_kernel)
        # 3. multitask kernel
        if args.gp.kernel == 'multitask': # medium flexible
            self.covar_module = gpytorch.kernels.MultitaskKernel(
                scaled_grid_kernel, num_tasks=num_task, rank=num_task 
            )
        elif args.gp.kernel == 'LcMRTF':  # higher flexible
            self.covar_module = gpytorch.kernels.LCMKernel(
                scaled_grid_kernel, num_tasks=num_task, rank=num_task 
            )
        else:
            raise NotImplementedError("This kernel type is not yet implemented.")
        """
        input_dims = train_x.shape[-1]
        # Mean function
        self.mean_module = gpytorch.means.MultitaskMean(
            ConstantMean(), num_tasks=num_task
        )
        
        self.base_covar_module = RBFKernel() # TODO: matern kernel (shared? not?)
        product_kernel = ProductStructureKernel(
            ScaleKernel(
                GridInterpolationKernel(self.base_covar_module, grid_size=20, num_dims=1, grid_bounds=[(-1.2, 1.2)])
                #TODO: grid_bounds => time [-0.2, 1.2]
            ), num_dims=input_dims
        )
        self.covar_module = MultitaskKernel(
            product_kernel,
            num_tasks=num_task,
            rank=num_task)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

    @staticmethod
    def init_from_data(args, prefix, train_x, train_y, likelihood):
        if prefix == 'transls': 
            num_task = 3
        elif prefix == 'rots': 
            num_task = get_rots_dim(args.motion.rot_type)
        return MultitaskGPModel(args, train_x, train_y, likelihood, num_task)

    @staticmethod
    def get_likelihood(args, prefix):
        if prefix == 'transls': 
            num_task = 3
            likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_task)

        elif prefix == 'rots':
            num_task = get_rots_dim(args.motion.rot_type)
            # TODO: changed for hyperspace
            likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_task)

        return likelihood

    @staticmethod
    def get_mll(likelihood, model):
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        return mll

    @staticmethod
    def get_optimizer(args, prefix, likelihood, model):
        lr = getattr(args.gp, f"{prefix}_gp_lr")
        return torch.optim.Adam([
                    {'params': model.parameters()},
                    ], lr=lr) 

############################################################################################
# Exact + Deep kernel
############################################################################################





############################################################################################
# Variational
############################################################################################







############################################################################################
# Variational + Deep Kernel
############################################################################################