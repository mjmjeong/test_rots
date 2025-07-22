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

from gpytorch.models import VariationalGP
from gpytorch.variational import CholeskyVariationalDistribution, MeanFieldVariationalDistribution, VariationalStrategy, LMCVariationalStrategy
from gpytorch.means import MultitaskMean, ConstantMean
from gpytorch.kernels import MultitaskKernel, RBFKernel, ScaleKernel, MaternKernel, ProductKernel
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.mlls import VariationalELBO
from gpytorch.variational.nearest_neighbor_variational_strategy import NNVariationalStrategy

from gp_module.kernel import *
from gp_module.utils import *
import numpy as np

from flow3d.transforms import get_rots_dim

############################################################################################
# Exact
############################################################################################
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, args, train_x, train_y, likelihood, num_task):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
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
        
        # TODO: add learnable / set length_scale init
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
    def get_mll(likelihood, model, train_x):
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
class MultitaskVariationalGPModel(ApproximateGP):
    """
    def __init__(self, args, inducing_points, num_tasks):
        self.args = args
        
        # TODO: d
#        variational_distribution = MeanFieldVariationalDistribution(inducing_points.size(0))        
        # Use NearestNeighborVariationalStrategy for grid-structured data
 #       variational_strategy = NNVariationalStrategy(
 #           self,
 #           inducing_points=inducing_points,
 #           variational_distribution=variational_distribution,
 #           k=15,  # Increased for spatial neighbors
 #       )
        # N

        num_latents = num_tasks
        batch_shape = torch.Size([num_latents])
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        multitask_strategy = gpytorch.variational.LMCVariationalStrategy(variational_strategy, num_tasks=num_tasks, num_latents=num_tasks)
        super().__init__(multitask_strategy)
        
        self.mean_module = gpytorch.means.ConstantMean()

        # Option: input kernel
        self.covar_module = ScaleKernel(
            MaternKernel(nu=1.5, ard_num_dims=inducing_points.size(-1))
        )

        if args.use_separable_kernel:
            spatial_base_kernel = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=3,active_dims=[0, 1, 2])
            spatial_kernel_scaled = gpytorch.kernels.ScaleKernel(spatial_base_kernel)

            temporal_base_kernel = gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=1,active_dims=[3])
            temporal_kernel_scaled = gpytorch.kernels.ScaleKernel(temporal_base_kernel)
            base_kernel = gpytorch.kernels.ProductKernel(
                spatial_kernel_scaled,
                temporal_kernel_scaled
            )
        else:
            base_kernel = ScaleKernel(
                MaternKernel(nu=2.5, ard_num_dims=4) 
            )
        
        # Option: grid
        if args.use_grid_kernel:
            grid_kernel = ProductStructureKernel(
                    ScaleKernel(
                    GridInterpolationKernel(base_kernel, grid_size=20, num_dims=1, grid_bounds=[(-1.2, 1.2)])
                ), num_dims=input_dims
                )
        else:
            grid_kernel = base_kernel

        # Option: output (multitask)
        if args.use_multitask:
            self.covar_module = gpytorch.kernels.MultitaskKernel(
                grid_kernel,
                num_tasks=num_tasks,
                rank=num_tasks  # TODO: Slightly higher rank for spatial correlations
            )
    """
    
    def __init__(self, args, inducing_points, num_tasks):
        self.args = args
        num_latents = num_tasks
        batch_shape = torch.Size([num_latents])
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0), 
            batch_shape=batch_shape
        )
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        
        multitask_strategy = LMCVariationalStrategy(
            variational_strategy, 
            num_tasks=num_tasks, 
            num_latents=num_latents
        )
        super().__init__(multitask_strategy)
        
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)
        self.covar_module = ScaleKernel(
            MaternKernel(nu=1.5, ard_num_dims=inducing_points.size(-1), batch_shape=batch_shape),
            batch_shape=batch_shape
        )
        self._initialize_hyperparameters(inducing_points)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

     # Initialize hyperparameters
    
    def _initialize_hyperparameters(self, inducing_points):
        """Initialize kernel hyperparameters based on data statistics"""
        
        # Calculate reasonable initial length scales based on inducing points
        input_dim = inducing_points.size(-1)
        
        with torch.no_grad():
            # Compute data ranges for each dimension
            data_ranges = inducing_points.max(dim=0)[0] - inducing_points.min(dim=0)[0]
            
            # Set initial length scales to be a fraction of the data range
            # This is a common heuristic: start with length scales around 10-50% of data range
            initial_lengthscales = data_ranges * 0.2  # 20% of data range
            
            # Ensure minimum length scale to avoid numerical issues
            initial_lengthscales = torch.ones_like(initial_lengthscales) * 0.01
            
            # Set length scales for each latent function
            if hasattr(self.covar_module.base_kernel, 'lengthscale'):
                # For ARD kernel, set individual length scales
                num_latents = self.covar_module.base_kernel.batch_shape[0]
                lengthscales = initial_lengthscales.unsqueeze(0).repeat(num_latents, 1)
                self.covar_module.base_kernel.lengthscale = lengthscales
            
            # Initialize output scales (variance)
            # Start with moderate values, not too small or too large
            initial_outputscale = torch.ones(self.covar_module.batch_shape) * 1.0
            self.covar_module.outputscale = initial_outputscale
            
            # Initialize mean to zero (already default, but explicit)
            if hasattr(self.mean_module, 'constant'):
                self.mean_module.constant.fill_(0.0)
        
        print(f"Initialized length scales: {self.covar_module.base_kernel.lengthscale}")
        print(f"Initialized output scales: {self.covar_module.outputscale}")
        print(f"Initialized means: {self.mean_module.constant if hasattr(self.mean_module, 'constant') else 'N/A'}")


    @staticmethod
    def init_from_data(args, prefix, train_x, train_y, likelihood):
        if prefix == 'transls': 
            num_tasks = 3
        elif prefix == 'rots': 
            num_tasks = get_rots_dim(args.motion.rot_type)

        inducing_points = create_adaptive_inducing_points(train_x, num_inducing=args.gp.inducing_num, 
                                                        method=args.gp.inducing_method, 
                                                        inducing_min=args.gp.inducing_min,
                                                        inducing_max=args.gp.inducing_max,
                                                        add_noise_scale=args.gp.inducing_point_noise_scale)

        return MultitaskVariationalGPModel(args.gp, inducing_points, num_tasks)
            
    @staticmethod
    def get_likelihood(args, prefix):
        if prefix == 'transls': 
            num_tasks = 3            
            likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks)

        elif prefix == 'rots':
            num_tasks = get_rots_dim(args.motion.rot_type)
            likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks)

        return likelihood

    @staticmethod
    def get_mll(likelihood, model, train_x):
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_x.size(0))
        return mll

    @staticmethod
    def get_optimizer(args, prefix, likelihood, model):
        lr = getattr(args.gp, f"{prefix}_gp_lr")
        return torch.optim.Adam([
                    {'params': model.parameters()},
                    {'params': likelihood.parameters()},
                    ], lr=lr) 

class IndependentVariationalGPModel(ApproximateGP):

    @staticmethod
    def init_from_data(args, prefix, train_x, train_y, likelihood, others=None):
        if prefix == 'transls': 
            num_tasks = 3
        elif prefix == 'rots': 
            num_tasks = get_rots_dim(args.motion.rot_type)

        inducing_points = create_adaptive_inducing_points(train_x, others, args)
        train_x = inducing_points

        print(f"----------------inducing points stats: {prefix}-----------------------")
        print("Data shape:", train_x.shape)
        print("Unique points:", torch.unique(train_x, dim=0).shape[0])

        # Check for numerical issues
        print("Data range:", train_x.min(), train_x.max())
        print("Data std:", train_x.std())

        # Check for exact duplicates
        unique_points = torch.unique(train_x, dim=0)
        print(f"Original points: {train_x.shape[0]}")
        print(f"Unique points: {unique_points.shape[0]}")

        # Check for near-duplicates (within 1e-6)
        from torch import cdist
        distances = torch.cdist(inducing_points.cpu(), inducing_points.cpu())
        near_duplicates = (distances < 1e-6) & (distances > 0)
        if near_duplicates.any():
            print(f"Found {near_duplicates.sum()} near-duplicate pairs")
        print("-------------------------------------------------------------")
                    
        if args.gp.inducing_task_specific:
            inducing_points = inducing_points.unsqueeze(0)
            inducing_points = inducing_points.repeat(num_tasks, 1, 1)  
        return IndependentVariationalGPModel(args, inducing_points, num_tasks, prefix)  

    def __init__(self, args, inducing_points, num_tasks, prefix):
        
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_tasks])
        )

        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ),
            num_tasks=num_tasks,
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_tasks]))

        nu_xy = getattr(args.gp, "nu_matern_xy")
        nu_zt = getattr(args.gp, "nu_matern_zt")
        initial_lengthscale_xy = getattr(args.gp, f"{prefix}_lengthscale_xy")
        initial_lengthscale_zt = getattr(args.gp, f"{prefix}_lengthscale_zt")

        self.covar_module = HexplaneMaternKernel(nus=[nu_xy, nu_zt], 
                                                batch_shape=torch.Size([num_tasks]), 
                                                combine=args.gp.combine_type,
                                                initial_lengthscale_xy=initial_lengthscale_xy,
                                                initial_lengthscale_zt=initial_lengthscale_zt,)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    @staticmethod
    def get_likelihood(args, prefix):
        if prefix == 'transls': 
            num_tasks = 3            
            likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)

        elif prefix == 'rots':
            num_tasks = get_rots_dim(args.motion.rot_type)
            likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)
        return likelihood

    @staticmethod
    def get_mll(likelihood, model, train_x):
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_x.size(0))
        return mll

    @staticmethod
    def get_optimizer(args, prefix, likelihood, model):
        lr = getattr(args.gp, f"{prefix}_gp_lr")
        return torch.optim.Adam([
                    {'params': model.parameters()},
                    {'params': likelihood.parameters()},
                    ], lr=lr) 

############################################################################################
# Variational + Deep Kernel + independent kernel
############################################################################################

class DeepMultitaskVariationalGP(ApproximateGP):
    def __init__(self, args, inducing_points, num_tasks):
        pass
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

    @staticmethod
    def init_from_data(args, prefix, train_x, train_y, likelihood):
        if prefix == 'transls': 
            num_tasks = 3
        elif prefix == 'rots': 
            num_tasks = get_rots_dim(args.motion.rot_type)

        inducing_points = create_adaptive_inducing_points(train_x, num_inducing=512, method='grid')

        return MultitaskVariationalGP(args.gp, inducing_points, num_tasks)
            
    @staticmethod
    def get_likelihood(args, prefix):
        if prefix == 'transls': 
            num_tasks = 3            
            likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks)

        elif prefix == 'rots':
            num_tasks = get_rots_dim(args.motion.rot_type)
            # TODO: changed for hyperspace
            likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks)
        return likelihood

    @staticmethod
    def get_mll(likelihood, model, train_x):
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_x.size(0))
        return mll

    @staticmethod
    def get_optimizer(args, prefix, likelihood, model):
        lr = getattr(args.gp, f"{prefix}_gp_lr")
        return torch.optim.Adam([
                    {'params': model.parameters()},
                    {'params': likelihood.parameters()},
                    ], lr=lr) 

##############################################################################
# Embedder
##############################################################################
# small MLP + VGP / Optimization / Drawing => TOD
# 1) positional encoding
# 2) 

"""
class MLP(nn.Module)
    def __init__()

        pass

    def positional_encoding(self)
        return

    @staticmethod
    def init():
        pass
"""
