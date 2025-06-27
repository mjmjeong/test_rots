import torch
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt
import numpy as np


from gp_module.model import *
from gp_module.kernel import *
from gp_module.likelihood import *
from gp_module.loss import *
from gp_module.uncertainty import *
from gp_module.utils import *


class Motion_GP():
    def __init__(self, args):
        self.args = args
        self.device = args.device
        
        self.transls_gp = eval(self.args.gp.transls_model)
        self.rots_gp = eval(self.args.gp.rots_model)

    def get_dataset(self, transls, rots,  canonical_idx=0, confidence=None):
        self.set_bbox(transls, 'transls')
        transls = self.bbox_normalize(transls, 'transls')
        self.set_bbox(rots, 'rots')
        rots = self.bbox_normalize(rots, 'rots')
        
        num_data_points = transls.shape[0]
        total_time = transls.shape[1] # Time
        # X0, Y0, Z0 
        can_xyz = transls[:,canonical_idx]
        can_xyz = can_xyz.unsqueeze(1).repeat(1, total_time, 1)
        # target time
        tgt_time = torch.linspace(-1, 1, total_time)
        tgt_time = tgt_time.view(1, total_time, 1).repeat(num_data_points, 1, 1)
        # C0 => rsampling
        can_confidence = confidence[:, canonical_idx]
        can_confidence = can_confidence.unsqueeze(1).repeat(1, total_time, 1)
        
        train_x = torch.cat((can_xyz, tgt_time, can_confidence), dim=-1)
        train_y = torch.cat((transls, rots, confidence), -1)
    
        self.viz_train_x = train_x[::10, :,:]
        self.viz_train_y = train_y[::10, :,:]

        train_x = train_x.reshape(num_data_points*total_time, -1)
        train_y = train_y.reshape(num_data_points*total_time, -1)

        confidence_mask = train_y[:,-1] > self.args.gp.confidence_thred

        train_x = train_x[confidence_mask]
        train_y = train_y[confidence_mask]

        return train_x, train_y
        
    def init_gp(self, train_x, train_y):

        train_y_transls = train_y[...,:3]
        train_y_rots = train_y[...,3:]

        # likelihood        
        self.transls_likelihood = self.transls_gp.get_likelihood(self.args, "transls")
        self.rots_likelihood = self.transls_gp.get_likelihood(self.args, "rots")

        # gp_model       
        self.transls_gp_model = self.transls_gp.init_from_data(self.args, 'transls', train_x, train_y_transls, self.transls_likelihood)
        self.rots_gp_model = self.rots_gp.init_from_data(self.args, 'rots', train_x, train_y_rots, self.rots_likelihood)

        # mll (loss)
        self.transls_mll = self.transls_gp.get_mll(self.transls_likelihood, self.transls_gp_model, train_x)
        self.rots_mll = self.rots_gp.get_mll(self.rots_likelihood, self.rots_gp_model, train_x)

        # optimizer
        self.transls_optimizer = self.transls_gp.get_optimizer(self.args, 'transls', self.transls_likelihood, self.transls_gp_model)
        self.rots_optimizer = self.rots_gp.get_optimizer(self.args, 'rots', self.rots_likelihood, self.rots_gp_model)   

    def set_bbox(self, data, prefix):
        min_vals = data.min(dim=0, keepdim=True).values.min(dim=1, keepdim=True).values.squeeze(0)
        max_vals = data.max(dim=0, keepdim=True).values.max(dim=1, keepdim=True).values.squeeze(0)

        min_for_scaling = torch.floor(min_vals)
        max_for_scaling = torch.ceil(max_vals)

        self.bbox = {f'{prefix}_min': min_for_scaling, f'{prefix}_max': max_for_scaling}
    
    def bbox_normalize(self, data, prefix):
        min_for_scaling = self.bbox[f'{prefix}_min']
        max_for_scaling = self.bbox[f'{prefix}_max']

        denominator = (max_for_scaling - min_for_scaling).unsqueeze(0)
        min_for_scaling_expanded = min_for_scaling.unsqueeze(0)

        data_scaled = 2 * (data - min_for_scaling_expanded) / denominator - 1
        return data_scaled

    def bbox_denormalize(self, normalized_data, prefix):
        
        min_for_scaling = self.bbox[f'{prefix}_min']
        max_for_scaling = self.bbox[f'{prefix}_max']

        denominator = (max_for_scaling - min_for_scaling).unsqueeze(0)
        min_for_scaling_expanded = min_for_scaling.unsqueeze(0)

        data = ((normalized_data + 1) /2.0)*denominator + min_for_scaling_expanded
        return data

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
    
    def set_device(self):
        self.transls_gp_model.to(self.device)
        self.transls_likelihood.to(self.device)
        self.rots_gp_model.to(self.device)
        self.rots_likelihood.to(self.device) 

    def set_mode(self, type_):
        if type_ == 'train':
            self.transls_gp_model.train()
            self.transls_likelihood.train()
            self.rots_gp_model.train()
            self.rots_likelihood.train()       

        elif type_ == 'eval':
            self.transls_gp_model.eval()
            self.transls_likelihood.eval()
            self.rots_gp_model.eval()
            self.rots_likelihood.eval()       

    def fitting_gp(self, train_x, train_y):
        if self.args.gp.transls_model in ['ExactGPModel']: 
            self.fitting_exact_gp(train_x, train_y) # full
        elif self.args.gp.transls_model in ['MultitaskVariationalGPModel', 'IndependentVariationalGPModel']:
            self.fitting_variational_gp(train_x, train_y)
        else:
            raise NotImplementedError("Check training loop selection")

    def fitting_exact_gp(self, train_x, train_y):
        # Initialization
        self.remove_gp()
        self.init_gp(train_x, train_y)
        
        self.set_device()
        self.set_mode('train')

        train_x = train_x.to(self.device)
        train_y = train_y.to(self.device)

        y_transls = train_y[:, :3]
        y_rots = train_y[:, 3:-1] 
        y_confidence = train_y[:, -1:]

        breakpoint()

        for i in range(self.args.gp.epochs):
            # reset gradient
            self.transls_optimizer.zero_grad()
            self.rots_optimizer.zero_grad()
            with gpytorch.settings.use_toeplitz(False), gpytorch.settings.max_root_decomposition_size(30):
                # forward
                print("train_x:", train_x.shape)
                transls_output = self.transls_gp_model(train_x)
                transls_loss = -1 * self.transls_mll(transls_output, y_transls)
                #, batch_confidence) 

                rots_output = self.rots_gp_model(train_x)
                rots_loss = -1 * self.rots_mll(rots_output, y_rots)
                #, batch_confidence)

                loss = transls_loss + rots_loss

                # backward
                loss.backward()
                self.transls_optimizer.step()
                self.rots_optimizer.step()

            if (i + 1) % 1 == 0:
                print(f"[{i+1}] Loss: {loss.item():.4f}")


    def fitting_variational_gp(self, train_x, train_y):

        train_x_confidence = train_x[..., -1:]
        train_x = train_x[..., :-1]

        train_y_confidence = train_y[..., -1:]
        train_y = train_y[..., :-1]
        
        # Initialization
        self.remove_gp()
        self.init_gp(train_x, train_y)

        self.set_device()
        self.set_mode('train')

        train_x = train_x.to(self.device)
        train_y = train_y.to(self.device)
        

        y_transls = train_y[:, :3]
        y_rots = train_y[:, 3:-1] 
        y_confidence = train_y[:, -1:]

        self.set_device()
        self.set_mode('train')

        # define dataloader
        train_dataset = TensorDataset(train_x, train_y, train_x_confidence, train_y_confidence)
        train_loader = DataLoader(train_dataset, batch_size=self.args.gp.batch_size, shuffle=True) 
        
        for i in range(self.args.gp.epochs):
            for batch_idx, (batch_x, batch_y, batch_x_conf, batch_y_conf) in enumerate(train_loader):
                # input x
                batch_x = batch_x.to(self.device)
                batch_x_before = batch_x
                if self.args.gp.x_rsample != 'none':
                    batch_x_stdv = estimate_std(batch_x, batch_x_conf.to(self.device), i, args=self.args.gp) 
                    batch_x = torch.distributions.Normal(batch_x, batch_x_stdv).rsample()
                # gt y
                batch_y_transls = batch_y[..., :3].to(self.device)
                batch_y_rots = batch_y[..., 3:].to(self.device)
                batch_y_confidence = batch_y_conf.to(self.device)

                # reset gradient
                self.transls_optimizer.zero_grad()
                self.rots_optimizer.zero_grad()
 
                # forward
                transls_output = self.transls_gp_model(batch_x)
                transls_loss = -1 * self.transls_mll(transls_output, batch_y_transls)
                #, batch_confidence) 
                # TODO: confidence aware MLL

                rots_output = self.rots_gp_model(batch_x)
                rots_loss = -1 * self.rots_mll(rots_output, batch_y_rots)
                #, batch_confidence)

                loss = transls_loss + rots_loss
                loss.backward()

                # TODO
                torch.nn.utils.clip_grad_norm_(self.transls_gp_model.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.rots_gp_model.parameters(), max_norm=1.0)

                self.transls_optimizer.step()
                self.rots_optimizer.step()

                
            if (i) % 1 == 0:
                print(f"[{i+1}] Loss: {loss.item():.4f}")
            
            if i % 100 == 0:
                print('save_image')
                self.plot_traj(self.viz_train_x, self.viz_train_y, name=self.args.exp_name, step=i//100, loss=loss.detach().item())

                    # Print some kernel parameters for monitoring
#                    if hasattr(model.covar_module.data_covar_module, 'base_kernel'):
#                        base_kernel = model.covar_module.data_covar_module.base_kernel
#                        if hasattr(base_kernel, 'kernels'):
                            # Separable kernel
#                            spatial_ls = base_kernel.kernels[0].base_kernel.lengthscale
#                            temporal_ls = base_kernel.kernels[1].base_kernel.lengthscale
#                            print(f'  Spatial lengthscales: {spatial_ls.detach().numpy()}')
#                            print(f'  Temporal lengthscale: {temporal_ls.detach().numpy()}')

    def sampling(self, data, keep=False):
        x = data
        transls_output = self.transls_gp_model(x)
        rots_output = self.rots_gp_model(x)
        return transls_output, rots_output

    def recon_loss(self, transls_curr, rots_curr, confidence):
        pass

    # utils
    def plot_traj(self, train_x, train_y, name=None, step=None, loss=None): 
        self.set_mode('eval')
        train_x = train_x.to(self.device)
        T = train_x.size(1)
        N = train_x.size(0)

        train_x = train_x[..., :-1]
        train_x = train_x.reshape(N*T, -1)
        transls_output, rots_output = self.sampling(train_x)
        pred_transls = transls_output.mean.reshape(-1,T,3).detach().cpu()
        tgt_transls = train_y[:,:,:3]

        # visualization
        grid_x = np.linspace(0,1,T)
        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        fig.suptitle(f'Tensor Visualization: Predicted vs Target: Step {step} Loss {loss} ', fontsize=16)

        # Dimension labels
        dim_labels = ['X', 'Y', 'Z']
        colors = plt.cm.hsv(np.linspace(0, 1, N))

        # Plot each dimension
        for dim in range(3):  # For X, Y, Z dimensions
            # Plot predicted values (left column)
            axes[dim, 0].set_title(f'Predicted - {dim_labels[dim]} dimension')
            for line_idx in range(N):
                axes[dim, 0].plot(grid_x, pred_transls[line_idx, :, dim], alpha=0.3, color=colors[line_idx])
            axes[dim, 0].set_xlabel('X (0-1)')
            axes[dim, 0].set_ylabel(f'{dim_labels[dim]} values')
            axes[dim, 0].grid(True, alpha=0.3)
            
            # Plot target values (right column)
            axes[dim, 1].set_title(f'Target - {dim_labels[dim]} dimension')
            for line_idx in range(N):
                axes[dim, 1].plot(grid_x, tgt_transls[line_idx, :, dim], alpha=0.3, color=colors[line_idx])
            axes[dim, 1].set_xlabel('X (0-1)')
            axes[dim, 1].set_ylabel(f'{dim_labels[dim]} values')
            axes[dim, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        if name is None:
            name = 'tensor_visualization'
        plt.savefig(f'{name}_{step}.png', dpi=300, bbox_inches='tight')
        plt.close()
        self.set_mode('train')