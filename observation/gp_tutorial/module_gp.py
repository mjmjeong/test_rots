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
            gp_model = LMC_MultitaskGPModel(4)
            likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)
            mll = gpytorch.mlls.ExactMarginlLogLikelihood(likelihood, model)

            optimizer = optimizer = torch.optim.Adam([
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
        # initialization
        self.remove_gp()            
        self.init_gp(transls, rots, confidence)

        # train mode
        self.transls_gp_model.train()
        self.transls_likelihood.train()
        self.rots_gp_model.train()
        self.rots_likelihood.train()


        # fitting
        for i in range(self.args.gp_epochs):
            self.transls_optimizer.zero_grad()
            self.rots_optimizer.zero_grad()
            
            #x_infer = torch.linspace(0, 1, self.T)
            x_infer = torch.linspace(0, 1, self.T).unsqueeze(-1).repeat(1, 3)
            # transls
            if self.args.gp_transls_type == 'VI':
                breakpoint()
                transls_gp_out = self.transls_gp_model(x_infer )
                breakpoint()
                loss = -1 * self.transls_mll(transls_gp_out, transls)                

            # rots
            if self.args.gp_transls_type == 'LMC':
                breakpoint( )
                random_idx = torch.randperm(len(train_x))
                train_x_i = train_x[random_idx][:args.batch_size]
                train_y_i = train_y[random_idx][:args.batch_size]
                indices_i = indices[random_idx][:args.batch_size]
                #confidence_mask_i = confidence_mask[random_idx][:args.batch_size]
                confidence_mask_i = torch.ones_like(train_y_i)
                
                # Pass through the model
                pred_mean, pred_covar = model(train_x_i, task_indices=indices_i)
                
                # Compute the negative log marginal likelihood (with weighted mask)
                # Mask/weighting the loss by confidence_mask_i
                loss = -mll(pred_mean, pred_covar, train_y_i, confidence_mask_i)                        
                pred = model(train_x_i, task_indices=indices_i)

            loss.backward()
            #print(f"[{i+1}] Loss: {loss.item():.4f}")
            optimizer.step()

    def eval_mode(self):
        self.transls_gp_model.eval()
        self.rots_gp_model.eval()

        # save
        x_infer = torch.linspace(0, 1, self.T)
        breakpoint()
        transls_output = self.transls_gp_model(x_infer.unsqueeze(-1).repeat(1,3)) 
        rots_output = self.rots_gp_model(x_infer)
        self.transls_from_gp = transls_output.mean.permute(1,0).unsqueeze(-1).repeat(1,1,3)
        self.rots_from_gp = rots.mean.permute(1,0).unsqueeze(-1).repeat(1,1,3)

        self.transls_var_from_gp = transls_output.var
        self.rots_var_from_gp = transls_output.var
        
    def update_with_densification(self, reducing_index):
        #
        # update the weight
        pass

    def recon_loss(self, data):
        time = T
        # get target data
        transls_gp = self.transls_from_gp[:,time]
        rots_gp = self.rots_from_gp[:,time]
        weight = self.uncertainty_from_gp[:, time] 

        transls_recon_loss = F.mse_loss(transls_gp, transls_curr, reduction=None)
        transls_recon_loss = (weight * transls_recon_loss).mean()

        rots_recon_loss = F.mse_loss(rots_gp, rots_curr, reduction=None)
        rots_recon_loss = (weight * rots_recon_loss).mean()

        return {
                "rot_recon_gp": rots_recon_loss,
                "transls_recon_gp": transls_recon_loss           
                }

    def plot_traj(transls, name=None):
        breakpoint()
        dim_labels = ['X', 'Y', 'Z']  # 마지막 dim의 이름

        # 하나의 큰 figure 생성
        plt.figure(figsize=(12, 12))

        for dim in range(3):
            plt.subplot(3, 1, dim+1)  # 3행 1열의 subplot에서 dim+1번째 위치
            for i in range(40):  # 40개의 샘플
                plt.plot(transls[i, :, dim].detach().cpu().numpy(), label=f'Sample {i}')
            plt.title(f'Translation - {dim_labels[dim]} axis')
            plt.xlabel('Time Step')
            plt.ylabel(f'{dim_labels[dim]} Value')
            #plt.legend()
            plt.grid(True)

        # 전체 레이아웃 조정
        plt.tight_layout()
        # 하나의 PNG 파일로 저장
        if name is None:
            name = 'translation_all_axes.png'
        else:
            name = f'{name}.png'
        plt.savefig(name, dpi=300)
        plt.close()  # 메모리 관리를 위해 figure 닫기 
        
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
        assert (confidence.size(0) == data.size(0))
        device = data.device

        # inducing point initialization
        num_inducing = args.num_inducing
        inducing_points = torch.linspace(0, 1, num_inducing).to(device)
        
        if args.inducing_share:
            inducing_points =  inducing_points.reshape(1, -1, 1).repeat(num_latent*output_dim, 1, 1)
        else:
            inducing_points =  inducing_points.reshape(1, -1, 1).repeat(num_latent*output_dim, 1) 

 
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
#            num_latents=num_latent,
            latent_dim=-1
        )

        super().__init__(variational_strategy)        
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)
        if getattr(args, f'{prefix}_kernel_type') == 1:
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(batch_shape=batch_shape, ard_num_dims=data.size(-1)),
                batch_shape=batch_shape,
            )
            self.covar_module.base_kernel.lengthscale = getattr(args, f'{prefix}_lengthscale') * torch.ones(num_latent, 1, data.size(-1))
        
        elif getattr(args, f'{prefix}_kernel_type') == 2:
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(batch_shape=batch_shape),
                batch_shape=batch_shape,
            )
            self.covar_module.base_kernel.lengthscale = getattr(args, f'{prefix}_lengthscale') * torch.ones(num_latent, 1, 1)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

#################################################################
# Kernel custom
#################################################################


#################################################################
# W-ELBO
#################################################################
class WeightedVariationalELBO(gpytorch.mlls.VariationalELBO):
    def forward(self, variational_dist, target, confidence_mask=None):
        # 기본 ELBO 계산
        elbo = super().forward(variational_dist, target)
        
        if confidence_mask is not None:
            confidence_mask = confidence_mask.view_as(elbo)
            weighted_elbo = elbo * confidence_mask
            return weighted_elbo.sum() / confidence_mask.sum()
        
        return elbo
