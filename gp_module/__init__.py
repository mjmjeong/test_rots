import torch
from torch.utils.data import TensorDataset, DataLoader
import time

import matplotlib.pyplot as plt
import numpy as np
import copy

from gp_module.model import *
from gp_module.kernel import *
from gp_module.likelihood import *
from gp_module.loss import *
from gp_module.uncertainty import *
from gp_module.utils import *

import csv
import os

class MotionGP():
    def __init__(self, args):

        self.args = args
        self.bbox_stats = {} 
        self.transls_gp = eval(self.args.gp.transls_model)
        self.rots_gp = eval(self.args.gp.rots_model)
        
        self.is_trained = False  # True = gp_model is optimized at least once
        self.canonical_frame_idx = 'inf'
        self.data_cano = None

    def get_gp_gs_inference_input(self, transls_all, tgt_gs):
        # TODO
        """
        input
        - transls_all: N,T,3
        - tgt_gs: [times] # T_tgt
        """
        #if self.input_feature_type == 'global_xyz':
        #    input_feature_xyz = cano_mean.unsqueeze(1).repeat(1, total_time, 1)
        #else self.input_feature_type == 'canonical_idx_diff': 
        #    breakpoint()
        breakpoint()
        if tgt_gs is not None:
            tgt_t_len = len(tgt_gs) # target time number
        else:
            tgt_t_len = self.bbox_stats['time'].size(1)
        G = len(transls_all) # gaussian number

        # transls
        can_transls = transls_all[:, self.canonical_frame_idx, :].unsqueeze(1)
        normed_can_transls = self.normalize_to_bbox(can_transls, prefix='transls')
        normed_can_transls = normed_can_transls.repeat(1, tgt_t_len, 1)
        
        # time
        if tgt_gs is not None: 
            normed_time = self.bbox_stats['time'][:, list(tgt_gs), :] # 1, T, 1
        else:
            normed_time = self.bbox_stats['time'] # 1, T, 1

        normed_time = normed_time.repeat(G,1,1)
        gs_input = torch.cat((normed_can_transls, normed_time), -1) 
        gs_input = gs_input.reshape(G*tgt_t_len, -1)
        return gs_input


    def get_training_dataset(self, cano_mean, cano_quat, means, quats, transls, rots, confidence_weights):
        # TODO: is_trained=True / False
        print(f"first step?: {not self.is_trained} - bbox is updated!")
        device = transls.device

        if not self.is_trained:
            print(f"first step!! bbox is updated!")
            assert self.canonical_frame_idx == 'inf'
            if self.args.gp.canonical_type == 'first_frame':
                self.canonical_frame_idx = 0
            elif self.args.gp.canonical_type == 'largest_confidence':
                high_conf_mask = confidence_weights > 0.5
                self.canonical_frame_idx = (high_conf_mask).sum(0).argmax().item()
            print("canonical idx: ", self.canonical_frame_idx)
            self.delta_cano = self.args.gp.delta_cano

        if self.args.gp.valid_can_thre > 0:
            breakpoint()
            # TODO: update during training
            valid_cano_mask = confidence_weights[:,self.canonical_frame_idx, 0]>valid_can_thre
            cano_mean = cano_mean[valid_cano_mask]
            cano_quat = cano_quat[valid_cano_mask]
            means = transls[valid_cano_mask]
            quats = rots[valid_cano_mask]
            transls = transls[valid_cano_mask]
            rots = rots[valid_cano_mask]
            
            if confidence_weights is not None:
                confidence_weights = confidence_weights[valid_cano_mask]

        if not self.is_trained: 
            self.set_bbox_stats(cano_mean, 'cano_mean')
            self.set_bbox_stats(cano_quat, 'cano_quat')
            self.set_bbox_stats(means, 'means')
            self.set_bbox_stats(quats, 'quats')
            self.set_bbox_stats(transls, 'transls')
            self.set_bbox_stats(rots, 'rots')

            print(self.bbox_stats)

        transls = self.normalize_to_bbox(transls, 'transls')
        rots = self.normalize_to_bbox(rots, 'rots')        
        means = self.normalize_to_bbox(means, 'means')
        quats = self.normalize_to_bbox(quats, 'quats')
        cano_mean = self.normalize_to_bbox(cano_mean, 'cano_mean')
        cano_quat = self.normalize_to_bbox(cano_quat, 'cano_quat')

        num_data_points = transls.shape[0]
        total_time = transls.shape[1] # Time
        ##########################################################
        # Input
        ###########################################################
        # input_feature 1: X0, Y0, Z0 
        if self.input_feature_type == 'global_xyz':
            input_feature_xyz = cano_mean.unsqueeze(1).repeat(1, total_time, 1)
#            self.transls_cano = transls[:, self.canonical_frame_idx, :].unsqueeze(1)
#            self.rots_cano = rots[:, self.canonical_frame_idx, :].unsqueeze(1)
#            input_feature_xyz = transls[:,self.canonical_frame_idx]
#            input_feature_xyz = input_feature_xyz.unsqueeze(1).repeat(1, total_time, 1)
        
        # input_feature 2: target time
        if not self.is_trained:
            tgt_time = torch.linspace(-1, 1, total_time).to(device).view(1, total_time, 1)
            self.bbox_stats['time'] = copy.deepcopy(tgt_time)
        else:
            tgt_time = self.bbox_stats['time']
        tgt_time = tgt_time.repeat(num_data_points, 1, 1)

        # C0 => rsampling
        if self.input_feature_type == 'global_xyz':
            canonical_confidence_weights = torch.ones_like(tgt_time)
        #elif self.input_feature_type == 'canonical_idx_diff':         
        #    canonical_confidence_weights = confidence_weights[:, self.canonical_frame_idx]
        #    canonical_confidence_weights = canonical_confidence_weights.unsqueeze(1).repeat(1, total_time, 1)

        breakpoint()
        input_features = torch.cat((can_xyz, tgt_time, canonical_confidence_weights), dim=-1)

        ##########################################################
        # output
        ###########################################################
        
        if self.output_feature_type == 'global_xyz':
            target_values = torch.cat((means, quats, confidence_weights), -1)
        elif self.output_feature_type == 'motion':
            target_values = torch.cat((transls, rots, confidence_weights), -1)
        elif self.output_feature_type == 'diff_xyz':
            target_values = torch.cat((means - cano_mean, quats - cano_quat, confidence_weights), -1)

        #if self.delta_cano:
        #    target_values = torch.cat((transls-self.transls_cano, rots-self.rots_cano, confidence_weights), -1)
        #else:

        if len(input_features) > 100:
            self.viz_input_features = input_features[::10, :,:][:10]
            self.viz_target_values = target_values[::10, :,:][:10]
        else:
            self.viz_input_features = input_features[:10, :,:]
            self.viz_target_values = target_values[:10, :,:]

        input_features = input_features.reshape(num_data_points*total_time, -1)
        target_values = target_values.reshape(num_data_points*total_time, -1)
        self.full_input_features = copy.deepcopy(input_features)

        confidence_mask = target_values[:,-1] > self.args.gp.confidence_thred

        input_features = input_features[confidence_mask]
        target_values = target_values[confidence_mask]
        
        return input_features, target_values
        
    def initialize_gp_models(self, input_features=None, target_values=None, others=None):

        target_values_transls = target_values[...,:3]
        target_values_rots = target_values[...,3:]

        # likelihood        
        self.transls_likelihood = self.transls_gp.get_likelihood(self.args, "transls")
        self.rots_likelihood = self.transls_gp.get_likelihood(self.args, "rots")

        # gp_model       
        self.transls_gp_model = self.transls_gp.init_from_data(self.args, 'transls', input_features, target_values_transls, self.transls_likelihood, others=others)
        self.rots_gp_model = self.rots_gp.init_from_data(self.args, 'rots', input_features, target_values_rots, self.rots_likelihood, others=others)

        # mll (loss)
        self.transls_mll = self.transls_gp.get_mll(self.transls_likelihood, self.transls_gp_model, input_features)
        self.rots_mll = self.rots_gp.get_mll(self.rots_likelihood, self.rots_gp_model, input_features)

        # optimizer
        self.transls_optimizer = self.transls_gp.get_optimizer(self.args, 'transls', self.transls_likelihood, self.transls_gp_model)
        self.rots_optimizer = self.rots_gp.get_optimizer(self.args, 'rots', self.rots_likelihood, self.rots_gp_model)

    def initialize_gp_models_from_state_dict(self, state_dict=None):
        self.is_trained = state_dict['is_trained']
        self.canonical_frame_idx = state_dict['canonical_frame_idx']
        self.bbox_stats = state_dict['bbox_stats']
        self.delta_cano = state_dict['delta_cano']

        # likelihood        
        self.transls_likelihood = self.transls_gp.get_likelihood(self.args, "transls")
        self.rots_likelihood = self.transls_gp.get_likelihood(self.args, "rots")
        self.transls_likelihood.load_state_dict(state_dict['transls_likelihood'])
        self.rots_likelihood.load_state_dict(state_dict['rots_likelihood'])

        # model
        inducing_key = [i for i in state_dict['transls_model'].keys() if i.endswith('inducing_points')][0]
        transls_inducing_points = state_dict['transls_model'][inducing_key]
        rots_inducing_points = state_dict['rots_model'][inducing_key]

        self.transls_gp_model = self.transls_gp.init_from_data(self.args, 'transls', train_x=None, train_y=None, 
                                                            likelihood=self.transls_likelihood, others=None,
                                                            inducing_points = transls_inducing_points)

        self.rots_gp_model = self.rots_gp.init_from_data(self.args, 'rots', train_x=None, train_y=None, 
                                                            likelihood=self.rots_likelihood, others=None,
                                                            inducing_points = rots_inducing_points)

        self.transls_gp_model.load_state_dict(state_dict['transls_model'])
        self.rots_gp_model.load_state_dict(state_dict['rots_model'])
        
        # optimizer
        self.transls_optimizer = self.transls_gp.get_optimizer(self.args, 'transls', self.transls_likelihood, self.transls_gp_model)
        self.rots_optimizer = self.rots_gp.get_optimizer(self.args, 'rots', self.rots_likelihood, self.rots_gp_model)   
        self.transls_optimizer.load_state_dict(state_dict['transls_optimizer'])
        self.rots_optimizer.load_state_dict(state_dict['rots_optimizer'])

        # should be update the number of data of MLL (loss)
        self.transls_mll = None
        self.rots_mll = None

    def reset_mll_data_num(self, new_input_features):
        self.transls_mll = self.transls_gp.get_mll(self.transls_likelihood, self.transls_gp_model, new_input_features)
        self.rots_mll = self.rots_gp.get_mll(self.rots_likelihood, self.rots_gp_model, new_input_features)

    def set_bbox_stats(self, data, prefix):
        min_vals = data.min(dim=0, keepdim=True).values.min(dim=1, keepdim=True).values.squeeze(0)
        max_vals = data.max(dim=0, keepdim=True).values.max(dim=1, keepdim=True).values.squeeze(0)

        min_for_scaling = torch.floor(min_vals)
        max_for_scaling = torch.ceil(max_vals)

        self.bbox_stats.update({f'{prefix}_min': min_for_scaling, f'{prefix}_max': max_for_scaling})
    
    def normalize_to_bbox(self, data, prefix):
        assert data.dim() == 3 
        min_for_scaling = self.bbox_stats[f'{prefix}_min']
        max_for_scaling = self.bbox_stats[f'{prefix}_max']

        denominator = (max_for_scaling - min_for_scaling).unsqueeze(0)
        min_for_scaling_expanded = min_for_scaling.unsqueeze(0)
        data_scaled = 2 * (data - min_for_scaling_expanded) / denominator - 1
        return data_scaled

    def denormalize_from_bbox(self, normalized_data, prefix):
        min_for_scaling = self.bbox_stats[f'{prefix}_min']
        max_for_scaling = self.bbox_stats[f'{prefix}_max']

        if normalized_data.dim() == 3:
            denominator = (max_for_scaling - min_for_scaling).unsqueeze(0)
            min_for_scaling_expanded = min_for_scaling.unsqueeze(0)

        elif normalized_data.dim() == 2: 
            denominator = (max_for_scaling - min_for_scaling)
            min_for_scaling_expanded = min_for_scaling
        data = ((normalized_data + 1) /2.0)*denominator + min_for_scaling_expanded
        return data

    def cleanup_gp_models(self):
        try:
            del self.transls_gp_model
            del self.rots_gp_model
            del self.transls_likelihood
            del self.rots_likelihood            
            del self.transls_optimizer
            del self.rots_optimizer
        except:
            pass
    
    def set_device(self, device):
        self.transls_gp_model.to(device)
        self.transls_likelihood.to(device)
        self.rots_gp_model.to(device)
        self.rots_likelihood.to(device) 

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

    def training_gp_models(self, input_features, target_values, skip_rots=False, prefix='', plot_graph=False):
        if self.is_trained:
            print("GP training is started for GP-GS medium step")
        else:
            print("GP training is started for the first step")
            
        if self.args.gp.transls_model in ['ExactGPModel']: 
            self.train_exact_gp(input_features, target_values, skip_rots=skip_rots, plot_graph=plot_graph) # full
        elif self.args.gp.transls_model in ['MultitaskVariationalGPModel', 'IndependentVariationalGPModel']:
            self.train_variational_gp(input_features, target_values, skip_rots=skip_rots, plot_graph=plot_graph)
        else:
            raise NotImplementedError("Check training loop selection")
        self.is_trained = True
        print("GP training is finished")
        
    def train_exact_gp(self, input_features, target_values):
        breakpoint()
        self.set_device()
        self.set_mode('train')

#        input_features = input_features.to(self.device)
#        target_values = target_values.to(self.device)

        target_transls = target_values[:, :3]
        target_rots = target_values[:, 3:-1] 
        target_confidence = target_values[:, -1:]

        for i in range(self.args.gp.inner_epochs):
            # reset gradient
            self.transls_optimizer.zero_grad()
            self.rots_optimizer.zero_grad()
            with gpytorch.settings.use_toeplitz(False), gpytorch.settings.max_root_decomposition_size(30):
                # forward
                print("input_features:", input_features.shape)
                transls_predicted = self.transls_gp_model(input_features)
                transls_loss = -1 * self.transls_mll(transls_predicted, target_transls)
                #, batch_confidence) 

                rots_predicted = self.rots_gp_model(input_features)
                rots_loss = -1 * self.rots_mll(rots_predicted, target_rots)
                #, batch_confidence)

                loss = transls_loss + rots_loss

                # backward
                loss.backward()
                self.transls_optimizer.step()
                self.rots_optimizer.step()

            if (i + 1) % 1 == 0:
                print(f"[{i+1}] Loss: {loss.item():.4f}")

    def train_variational_gp(self, input_features, target_values, skip_rots=False, plot_graph=False):
        self.optimization_stats = {} 

        input_confidence = input_features[..., -1:]
        input_features = input_features[..., :-1]

        target_confidence = target_values[..., -1:]
        target_values = target_values[..., :-1]

        device = input_features.device
        self.set_device(device)
        self.set_mode('train')

#        input_features = input_features.to(self.device)
#        target_values = target_values.to(self.device)

        # define dataloader
        training_dataset = TensorDataset(input_features, target_values, input_confidence, target_confidence)
        training_loader = DataLoader(training_dataset, batch_size=self.args.gp.inner_batch_size, shuffle=True)
        
        # TODO: check stability
        #gpytorch.settings.cholesky_jitter(1e-3)
        #gpytorch.settings.max_cholesky_size(2000)

        iteration_count = 0
        for epoch in range(self.args.gp.inner_epochs):
            if self.is_trained and iteration_count > self.args.gp.inner_iteration: 
                break
            for batch_idx, (batch_inputs, batch_targets, batch_input_conf, batch_target_conf) in enumerate(training_loader):
                # for GP-GS step 
                if self.is_trained: # TODO
                    print(f"this is not the first round: {iteration_count}")
                    if iteration_count > self.args.gp.inner_iteration:
                        break 
                batch_inputs = batch_inputs
                original_inputs = batch_inputs  # batch_x_before → original_inputs
        
                if self.args.gp.input_rsample != 'none':
                    input_stddev = estimate_std(batch_inputs, batch_input_conf, epoch, args=self.args.gp)
                    batch_inputs = torch.distributions.Normal(batch_inputs, input_stddev).rsample()
                
                # gt y
                batch_target_transls = batch_targets[..., :3]
                batch_target_rots = batch_targets[..., 3:]
                batch_target_confidence = batch_target_conf

                # reset gradient
                self.transls_optimizer.zero_grad()
                self.rots_optimizer.zero_grad()

                transls_predicted = self.transls_gp_model(batch_inputs)
                transls_loss = -1 * self.transls_mll(transls_predicted, batch_target_transls)
                #, batch_target_confidence) 
                # TODO: confidence aware MLL

                rots_predicted = self.rots_gp_model(batch_inputs)
                rots_loss = -1 * self.rots_mll(rots_predicted, batch_target_rots)
                #, batch_target_confidence)

                if skip_rots:
                    loss = transls_loss
                else:
                    loss = transls_loss + rots_loss
                loss.backward()

                # TODO
                torch.nn.utils.clip_grad_norm_(self.transls_gp_model.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.rots_gp_model.parameters(), max_norm=1.0)

                self.transls_optimizer.step()
                self.rots_optimizer.step()
            
                iteration_count += 1
            if epoch % 1 == 0:
                print(f"[{epoch+1}] Loss: {loss.item():.4f}")

        self.optimization_stats['loss'] = loss.item()
        self.optimization_stats['transls_xy_lengthscale'] = self.transls_gp_model.covar_module.get_average_lengthscale('xy')
        self.optimization_stats['transls_zt_lengthscale'] = self.transls_gp_model.covar_module.get_average_lengthscale('zt')
        
        if plot_graph:
            self.plot_traj(self.viz_input_features, self.viz_target_values, name=self.args.exp_name, step=epoch, optimization_stats=self.optimization_stats)

    def sampling(self, data, chunk_size=None, denorm=False):
        x = data
        if chunk_size is not None:
            total_step = (len(data) //chunk_size)+1
            transls_output, rots_output = [], []
            for step in range(total_step):
                try:
                    x = data[step*chunk_size:(step+1)*chunk_size]
                except:
                    x = data[step*chunk_size:]
                transls_output.append(self.transls_gp_model(x).mean.cpu())
                rots_output.append(self.rots_gp_model(x).mean.cpu())
            transls_output = torch.cat(transls_output, 0)
            rots_output = torch.cat(rots_output, 0)
        else:
            transls_output = self.transls_gp_model(x).mean
            rots_output = self.rots_gp_model(x).mean
        
        breakpoint() # regarding type
        if denorm:
            if self.delta_cano:
                transls_output += self.transls_cano
                rots_output += self.rots_cano

            transls_output = self.denormalize_from_bbox(transls_output, 'transls')
            rots_output = self.denormalize_from_bbox(rots_output, 'rots')
        return transls_output, rots_output
    
    def get_guidance(self, data, gaussian_num=None, chunk_size=None, denorm=False, skip_rots=False):
        transls_output_mean, rots_output_mean = [], []
        transls_output_var, rots_output_var = [], []

        with torch.no_grad():
            if chunk_size is not None and chunk_size > 0:
                total_step = (len(data) //chunk_size)+1

                for step in range(total_step):
                    try:
                        x = data[step*chunk_size:(step+1)*chunk_size]
                    except:
                        x = data[step*chunk_size:]
                    transls_output_mean.append(self.transls_gp_model(x).mean)
                    transls_output_var.append(self.transls_gp_model(x).variance)
                    if not skip_rots:
                        rots_output_mean.append(self.rots_gp_model(x).mean)
                        rots_output_var.append(self.rots_gp_model(x).variance)

                transls_output_mean = torch.cat(transls_output_mean, 0)
                transls_output_var = torch.cat(transls_output_var, 0)
                if not skip_rots:
                    rots_output_mean = torch.cat(rots_output_mean, 0)
                    rots_output_var = torch.cat(rots_output_var, 0)
            else:
                transls_output_mean = self.transls_gp_model(data).mean
                transls_output_var = self.transls_gp_model(data).var
                if not skip_rots:
                    rots_output_mean = self.rots_gp_model(data).mean
                    rots_output_var = self.rots_gp_model(data).var

            breakpoint() # following output type
            if denorm:
                breakpoint()
                if self.delta_cano:
                    transls_output_mean += self.transls_cano
                    if not skip_rots:
                        rots_output_mean += self.rots_cano
                transls_output_mean = self.denormalize_from_bbox(transls_output_mean, 'transls')
                if not skip_rots:
                    rots_output_mean = self.denormalize_from_bbox(rots_output_mean, 'rots')

        breakpoint()
        if gaussian_num is not None: 
            transls_output_mean = transls_output_mean.reshape(gaussian_num, -1, 3)
            transls_output_var = transls_output_var.reshape(gaussian_num, -1, 3)
        return transls_output_mean, transls_output_var, rots_output_mean, rots_output_var

    # utils
    def plot_traj(self, input_features, target_values, name=None, step=None, optimization_stats=None): 
        self.set_mode('eval')
        loss = optimization_stats['loss']
        
#        input_features = input_features.to(self.device)
        T = input_features.size(1)
        N = input_features.size(0)

        input_features = input_features[..., :-1]
        input_features = input_features.reshape(N*T, -1)
        transls_output, rots_output = self.sampling(input_features, chunk_size=5000)
        pred_transls = transls_output.reshape(-1,T,3).detach().cpu()
        tgt_transls = target_values[:,:,:3]

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
        plt.savefig(f'{name}.png', dpi=300, bbox_inches='tight')
        breakpoint()
        print(f"save image in {name}")
        plt.close()
        self.set_mode('train')

    def save_csv(self, filename="output.csv"):
        save_dict = {"data": self.args.init_data,
                    "data_num": self.args.data_num,
                    } 
        optimization_stats = copy.deepcopy(self.optimization_stats)
        save_dict.update(optimization_stats)
        update_dict = self.args.gp.__dict__
        save_dict.update(update_dict)
        
        file_exists = os.path.exists(filename)
        new_keys = list(save_dict.keys())

        if not file_exists:
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=new_keys)
                writer.writeheader()
                writer.writerow(save_dict)
        else:
            with open(filename, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                existing_header = reader.fieldnames
                existing_data = list(reader)

            missing_keys = [key for key in new_keys if key not in existing_header]
            
            if missing_keys:
                print(f"Adding new columns: {missing_keys}")
                updated_fieldnames = existing_header + missing_keys
                
                with open(filename, 'w', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=updated_fieldnames)
                    writer.writeheader()
                    
                    for row in existing_data:
                        writer.writerow(row)
                    
                    writer.writerow(save_dict)
            else:
                # No new columns, just append
                with open(filename, 'a', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=existing_header)
                    writer.writerow(save_dict)

    def get_state_dict(self):
        # transls: gp_model, optimizer, schedular
        # rots: gp_model, optimizer, schedular
        # is_ready: is this optimized at least once? 
        ckpt_dicts = {}
        ckpt_dicts['bbox_stats'] = self.bbox_stats
        ckpt_dicts['is_trained'] = self.is_trained
        ckpt_dicts['delta_cano'] = self.delta_cano
        ckpt_dicts['canonical_frame_idx'] = self.canonical_frame_idx

        ckpt_dicts['transls_model'] = self.transls_gp_model.state_dict()
        ckpt_dicts['rots_model'] = self.rots_gp_model.state_dict()

        ckpt_dicts['transls_likelihood'] = self.transls_likelihood.state_dict()
        ckpt_dicts['rots_likelihood'] = self.rots_likelihood.state_dict()

        ckpt_dicts['transls_optimizer'] = self.transls_optimizer.state_dict()
        ckpt_dicts['rots_optimizer'] = self.rots_optimizer.state_dict()

        return ckpt_dicts