import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import gpytorch

from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.models import ApproximateGP
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ZeroMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import VariationalELBO

from gpytorch.distributions import MultivariateNormal
import torch


"""
- training: visualization / optimization / kerenl selection
- exp 2: quaternion-baseline => diff smoothness
- exp 
- NGD: TODO
- Importance sampling: TODO
- Initialization - fitting: TODO (must be done today!!) 
- Importance sampling + Kernel fitting (modeling)
"""

#################################################################
# Unified model: training, update, loss
#################################################################
class Motion_GP():
    def __init__(self, args):
        self.args = args


    def init_gp(self, transls, rots, confidence):
        args = self.args
        num_data = rots.size(0)
        num_tasks = self.args.num_tasks
        T = rots.size(1)
        self.T = T
        assert (rots.size(0) == transls.size(0))
        assert (rots.size(0) == confidence.size(0))

        def define_exact(args, data, confidence, prefix):
            gp_model = LMC_MultitaskGPModel(data.size(-1))
            likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=data.size(-1))
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)
            optimizer = torch.optim.Adam([
                {'params': gp_model.parameters()},
                {'params': likelihood.parameters()},
            ], lr=0.001)
            return gp_model, likelihood, mll, optimizer
        
        def define_VI(args, data, confidence, prefix):
            num_data = data.size(0)
            assert(data.size(0) == confidence.size(0))

            gp_model = LMC_GPModel(args, data, confidence, prefix)
            likelihood = GaussianLikelihood(batch_shape=torch.Size([num_data]))
            mll = WeightedVariationalELBO(
                likelihood, 
                gp_model, 
                num_data=data.size(0)
            )
            optimizer = optimizer = torch.optim.Adam([
                    {'params': gp_model.parameters()},
                    {'params': likelihood.parameters()},
                            ], lr=getattr(args, f'{prefix}_gp_lr'))
            return gp_model, likelihood, mll, optimizer

        # transls
        if self.args.gp_transls_type == 'exactGP':
            self.transls_gp_model, self.transls_likelihood, self.transls_mll, self.transls_optimizer = define_exact(  
            args, transls, confidence, prefix='transls')
        elif self.args.gp_transls_type == 'VI':
            self.transls_gp_model, self.transls_likelihood, self.transls_mll, self.transls_optimizer = define_VI(args, transls, confidence, prefix='transls')

        # rots
        if self.args.gp_rots_type == 'exactGP':
            self.rots_gp_model, self.rots_likelihood, self.rots_mll, self.rots_optimizer = define_exact(args, rots, confidence, prefix='rots')
        elif self.args.gp_transls_type == 'VI':
            self.rots_gp_model, self.rots_likelihood, self.rots_mll, self.rots_optimizer = define_VI(args, rots, confidence, prefix='rots')

    def remove_gp(self):
        try:
            del self.transls_gp_model
            del self.rots_gp_model
            del self.transls_likelihood
            del self.rots_likelihood
            
            del self.transls_optimizer
            del self.rots_optimizer
        except:
            pass

    def fitting_gp(self, transls, rots, confidence):
        # Initialization
        self.remove_gp()
        self.init_gp(transls, rots, confidence)

        # Train mode
        self.transls_gp_model.train()
        self.transls_likelihood.train()
        self.rots_gp_model.train()
        self.rots_likelihood.train()

        # Prepare data
        num_data = transls.size(0)
        T = self.T
        x_infer = torch.linspace(0, 1, T).to(transls.device).unsqueeze(-1)  # Shape: [T, 1]

        # Fitting loop
        for i in range(self.args.gp_epochs):
            self.transls_optimizer.zero_grad()
            self.rots_optimizer.zero_grad()

            # Translations
            if self.args.gp_transls_type == 'VI':
                transls_output = self.transls_gp_model(x_infer)
                transls_loss = self.transls_mll(transls_output, transls) * confidence
                transls_loss = transls_loss.mean()
            elif self.args.gp_transls_type == 'exactGP':
                transls_output = self.transls_gp_model(x_infer)
                transls_loss = self.transls_mll(transls_output, transls)

            # Rotations
            if self.args.gp_rots_type == 'VI':
                rots_output = self.rots_gp_model(x_infer)
                rots_loss = -self.rots_mll(rots_output, rots) * confidence
                rots_loss = rots_loss.mean()
            elif self.args.gp_rots_type == 'exactGP':
                rots_output = self.rots_gp_model(x_infer)
                rots_loss = -self.rots_mll(rots_output, rots)

            # Combine losses
            loss = transls_loss + rots_loss
            loss.backward()
            self.transls_optimizer.step()
            self.rots_optimizer.step()

            if (i + 1) % 10 == 0:
                print(f"[{i+1}] Loss: {loss.item():.4f}")


    def eval_mode(self):
        self.transls_gp_model.eval()
        self.transls_likelihood.eval()
        self.rots_gp_model.eval()
        self.rots_likelihood.eval()

        # Inference
        x_infer = torch.linspace(0, 1, self.T).to(self.transls_gp_model.covar_module.base_kernel.lengthscale.device).unsqueeze(-1)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            transls_output = self.transls_likelihood(self.transls_gp_model(x_infer))
            rots_output = self.rots_likelihood(self.rots_gp_model(x_infer))

        # Store results
        self.transls_from_gp = transls_output.mean.view(self.T, -1, 3)  # Shape: [T, N, 3]
        self.rots_from_gp = rots_output.mean.view(self.T, -1, 3)  # Adjust based on rotation representation
        self.transls_var_from_gp = transls_output.variance.view(self.T, -1, 3)
        self.rots_var_from_gp = rots_output.variance.view(self.T, -1, 3)    
        
    def update_with_densification(self, reducing_index):
        #
        # update the weight
        pass

    def recon_loss(self, transls_curr, rots_curr, confidence):
        time = self.T - 1  # Use last time step or adjust as needed
        transls_gp = self.transls_from_gp[:, time]
        rots_gp = self.rots_from_gp[:, time]
        weight = confidence[:, time]

        transls_recon_loss = F.mse_loss(transls_gp, transls_curr, reduction='none')
        transls_recon_loss = (weight.unsqueeze(-1) * transls_recon_loss).mean()

        rots_recon_loss = F.mse_loss(rots_gp, rots_curr, reduction='none')
        rots_recon_loss = (weight.unsqueeze(-1) * rots_recon_loss).mean()

        return {
            "rot_recon_gp": rots_recon_loss,
            "transls_recon_gp": transls_recon_loss
        }

    def plot_traj(self, transls, name=None):
        dim_labels = ['X', 'Y', 'Z']  # Labels for dimensions

        # Validate input shape
        if transls.dim() != 3 or transls.size(-1) != 3:
            raise ValueError(f"Expected transls shape [N, T, 3], got {transls.shape}")

        # Create a single figure
        plt.figure(figsize=(12, 12))

        for dim in range(3):
            plt.subplot(3, 1, dim + 1)  # 3 rows, 1 column
            for i in range(min(40, transls.size(0))):  # Limit to 40 samples or fewer
                plt.plot(transls[i, :, dim].detach().cpu().numpy(), label=f'Sample {i}')
            plt.title(f'Translation - {dim_labels[dim]} axis')
            plt.xlabel('Time Step')
            plt.ylabel(f'{dim_labels[dim]} Value')
            plt.grid(True)

        # Save plot
        plt.tight_layout()
        if name is None:
            name = 'translation_all_axes.png'
        else:
            name = f'{name}.png'
        plt.savefig(name, dpi=300)
        plt.close()
         
#################################################################
# GP model
#################################################################
class LMC_MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, output_dim):
        super(LMC_MultitaskGPModel, self).__init__(output_dim)
        
        num_tasks = output_dim
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=output_dim
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=output_dim, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

class LMC_GPModel(gpytorch.models.ApproximateGP):
    def __init__(self, args, data, confidence, prefix=None):
        num_latent = args.num_tasks
        num_data = data.size(0)
        output_dim = data.size(-1)
        batch_shape = torch.Size([num_latent, output_dim])
        device = data.device

        # Inducing points
        num_inducing = args.num_inducing
        inducing_points = torch.linspace(0, 1, num_inducing).to(device)
        if args.inducing_share:
            inducing_points = inducing_points.reshape(1, -1, 1).repeat(num_latent * output_dim, 1, 1)
        else:
            inducing_points = inducing_points.reshape(1, -1, 1).repeat(num_latent * output_dim, 1)

        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing, batch_shape=torch.Size([num_latent])
        )
        variational_strategy = gpytorch.variational.LMCVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self,
                inducing_points.view(num_latent * output_dim, num_inducing, 1),
                variational_distribution,
                learn_inducing_locations=True,
            ),
            num_tasks=output_dim,
            latent_dim=-1
        )

        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)
        if prefix == 'rots' and getattr(args, f'{prefix}_kernel_type') == 'quaternion':
            self.covar_module = gpytorch.kernels.ScaleKernel(
                QuaternionKernel(batch_shape=batch_shape),
                batch_shape=batch_shape,
            )
        else:
            kernel = gpytorch.kernels.RBFKernel(
                batch_shape=batch_shape,
                ard_num_dims=data.size(-1) if getattr(args, f'{prefix}_kernel_type') == 1 else None
            )
            self.covar_module = gpytorch.kernels.ScaleKernel(
                kernel,
                batch_shape=batch_shape,
            )
            self.covar_module.base_kernel.lengthscale = getattr(args, f'{prefix}_lengthscale') * torch.ones(
                num_latent, 1, data.size(-1) if getattr(args, f'{prefix}_kernel_type') == 1 else 1
            )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        
