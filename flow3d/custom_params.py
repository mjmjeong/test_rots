import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import copy

from flow3d.transforms import cont_6d_to_rmat, quat_to_rmat, quat_t_to_dq, dq_to_quat_t
from flow3d.loss_utils import kl_div_gaussian, kl_div_vonmise_fisher, reparameterize_vonmise_fisher

from flow3d.transforms import *


class BinghamMotionBases(nn.Module):
    def __init__(self, rots, transls, cfg=None):
        super().__init__()
        self.num_frames = rots.shape[1]
        self.num_bases = rots.shape[0]
        self.cfg = cfg
        self.rot_type = cfg.motion.rot_type
    
        assert check_bases_sizes(rots, transls, self.rot_type)
        self.params = nn.ParameterDict(
            {
                "rots": nn.Parameter(rots),
                "transls": nn.Parameter(transls),
            }
        )
        if cfg.motion.init_opt_with_bing:
            self.init_opt_with_bing = True
            self.init_bingham()
        else:
            self.init_opt_with_bing = False

    def init_bingham(self):
        rots = self.params["rots"]
        if rots.size(-1) == 6:
            rots = sixd_to_quat(rots)
        
        bingham_param = bingham_mat2vec10(quaternion_to_bingham_matrix(rots))
        self.params = nn.ParameterDict(
            {
                "rots": nn.Parameter(bingham_param),
#                "rots": nn.Parameter(rots),
                "transls":  self.params["transls"]
            }
        )

    @staticmethod
    def init_from_state_dict(state_dict, cfg, prefix="params."):
        param_keys = ["rots", "transls"]
        assert all(f"{prefix}{k}" in state_dict for k in param_keys)
        args = {k: state_dict[f"{prefix}{k}"] for k in param_keys}
        args['cfg'] = cfg
        return BinghamMotionBases(**args)

    def cache_curr_state(self):
        breakpoint() # Annealing for update
        param_keys = ["rots", "transls"]
        self.rot_cache = copy.deepcopy(self.params['rots'].detach())
        self.transls_cache = copy.deepcopy(self.params['transls'].detach())


    def compute_transforms_default(self, ts: torch.Tensor, coefs: torch.Tensor, train=False) -> torch.Tensor:
        """
        :param ts (B)
        :param coefs (G, K)
        returns transforms (G, B, 3, 4)
        """
        # motion combine
        if self.rot_type == "6d":
            transls = self.params["transls"][:, ts]  # (K, B, 3)
            rots = self.params["rots"][:, ts]  # (K, B, 6)
            transls = torch.einsum("pk,kni->pni", coefs, transls)
            rots = torch.einsum("pk,kni->pni", coefs, rots)  # (G, B, 6)
            rotmats = cont_6d_to_rmat(rots)  # (G, B, 3, 3)
            return torch.cat([rotmats, transls[..., None]], dim=-1), {}

        elif self.rot_type == "quat":
            transls = self.params["transls"][:, ts]  # (K, B, 3)
            transls = torch.einsum("pk,kni->pni", coefs, transls)
            quat = self.params["rots"][:, ts]  # (K, B, 4)
            #quat = self.align_quat_to_max_coef(quat, coefs)
            quat = torch.einsum("pk,pkni->pni", coefs, quat)  # (G, B, 4):
            rotmats = quat_to_rmat(quat)  # (K, B, 3, 3)
            return torch.cat([rotmats, transls[..., None]], dim=-1), {}
    
        elif self.rot_type == "dual_quat":
            transls = self.params["transls"][:, ts]  # (K, B, 3)
            quat = self.params["rots"][:, ts]  # (K, B, 4)
            dual_quat = quat_t_to_dq(quat, transls) # (G, K, 8)
            dual_quat = self.align_quat_to_max_coef(dual_quat, coefs)  # (G, K, B, 8)
            dual_quat = torch.einsum("pk,pkni->pni", coefs, dual_quat)
            quat, transls = dq_to_quat_t(dual_quat)
            rotmats = quat_to_rmat(quat)
            return torch.cat([rotmats, transls[..., None]], dim=-1), {}

    def compute_transforms(self, ts: torch.Tensor, coefs: torch.Tensor, d_quat: torch.Tensor, train=False) -> torch.Tensor:
        """
        :param ts (B)
        :param coefs (G, K)
        :param rots of gaussian: (G, T, 4)
        returns transforms (G, B, 3, 4)
        """
        if d_quat is None:
            return self.compute_transofrms_default(ts, coefs, train)

        # transls       
        transls = self.params["transls"][:, ts]  # (K, B, 3)
        transls = torch.einsum("pk,kni->pni", coefs, transls)
        
        # rots: pritmitive
        breakpoint()
        primitive_quat = d_quat[:, ts] #  G, T, 4  -> G, 4
        rotmats = quat_to_rmat(primitive_quat)

        # rots: bingham
        breakpoint()
        bingham_param = self.params["rots"][:, ts]  # (B, 10)
        loss_dict = compute_bingham_geomety(bingham_param)

        # smoothness
        loss_dict['bingham_smooth'] =  F.mse_loss(self.params["rots"][:,:-1]-self.params["rots"][:,1:])

        # coeff computing
        bingham_param = torch.einsum("pk,kni->pni", coefs, bingham)  # (G, B, 10)
        loss_dict['bingham_recon'] = bingham_loglikelihood(bingham_param.detach(), G_quat)
        loss_dict['bingham_commit'] = bingham_loglikelihood(bingham_param, G_quat.detach())
        
        return torch.cat([rotmats, transls[..., None]], dim=-1), loss_dict 

class BayesianMotionBases(nn.Module):
    def __init__(self, rots, transls, rots_var=None, transls_var=None, cfg=None):
        super().__init__()
        self.num_frames = rots.shape[1]
        self.num_bases = rots.shape[0]
        self.cfg = cfg
        self.rot_type = cfg.motion.rot_type            

        # TODO: set_differ with activation function
        if rots_var is None:
            rots_var = torch.ones_like(rots) * cfg.motion.rots_var_init_value
        if transls_var is None:
            transls_var = torch.ones_like(transls) * cfg.motion.transls_var_init_value
            
        assert check_bases_sizes(rots, transls, self.rot_type)
        self.params = nn.ParameterDict(
            {
                "rots": nn.Parameter(rots),
                "transls": nn.Parameter(transls),
                "rots_var": nn.Parameter(rots_var),
                "transls_var": nn.Parameter(transls_var),
            }
        )

        self.cache_curr_state()
        self.set_var_activation()
        self.set_kl_loss()

    @staticmethod
    def init_from_state_dict(state_dict, cfg, prefix="params."):
        param_keys = ["rots", "transls", 'rots_var', "transls_var"]
        assert all(f"{prefix}{k}" in state_dict for k in param_keys)
        args = {k: state_dict[f"{prefix}{k}"] for k in param_keys}
        args['cfg'] = cfg
        return BayesianMotionBases(**args)

    def cache_curr_state(self):
        self.prev_rots = copy.deepcopy(self.params.rots).detach()
        self.prev_transls = copy.deepcopy(self.params.transls).detach()
        self.prev_rots_var = copy.deepcopy(self.params.rots_var).detach()
        self.prev_transls_var = copy.deepcopy(self.params.transls_var).detach()     

    def set_var_activation(self):
        if self.cfg.motion.var_activation == 'exp':
            self.func_var_act = torch.exp
        elif self.cfg.motion.var_activation == 'softplus':
            self.func_var_act = F.softplus
        
    def set_kl_loss(self):
        self.func_kl_transls = kl_div_gaussian
        if self.rot_type == "6d":
            self.func_kl_rot = kl_div_gaussian
        elif self.rot_type == 'quat':
            self.func_kl_rot = kl_div_vonmise_fisher
        elif self.rot_type ==  'dual_quat':
            self.func_kl_rot = kl_div_vonmise_fisher
        elif self.rot_type == 'bingham':
            self.func_kl_rot == kl_div_vonmise_fisher
        
    ###############################################################
    # aggregation method
    ###############################################################
    def compute_transforms(self, ts: torch.Tensor, coefs: torch.Tensor, train=False) -> torch.Tensor:
        """
        :param ts (B)
        :param coefs (G, K)
        returns transforms (G, B, 3, 4)
        """
        loss_dict = {}
        # motion combine
        if self.rot_type == "6d":
            transls = self.params["transls"][:, ts]  # (K, B, 3)
            rots = self.params["rots"][:, ts]  # (K, B, 6)
            transls = torch.einsum("pk,kni->pni", coefs, transls)
            rots = torch.einsum("pk,kni->pni", coefs, rots)  # (G, B, 6)
            if train:
                transls_mean = transls
                rots_mean = rots
                rots_var = self.func_var_act(self.params["rots_var"][:, ts])
                transls_var = self.func_var_act(self.params["transls_var"][:, ts])
                transls_var = torch.einsum("pk,kni->pni", coefs**2, transls_var)
                rots_var = torch.einsum("pk,kni->pni", coefs**2, rots_var)  # (G, B, 6)
                rots = rots_mean + torch.sqrt(rots_var) * torch.randn_like(rots_mean)
                transls = transls_mean + torch.sqrt(transls_var) * torch.randn_like(transls_mean)                
            rotmats = cont_6d_to_rmat(rots)  # (G, B, 3, 3)

        elif self.rot_type == "quat":
            transls = self.params["transls"][:, ts]  # (K, B, 3)
            transls = torch.einsum("pk,kni->pni", coefs, transls)
            quat = self.params["rots"][:, ts]  # (K, B, 4)
            quat = self.align_quat_to_max_coef(quat, coefs)
            quat = torch.einsum("pk,pkni->pni", coefs, quat)  # (G, B, 4)
            rotmats = quat_to_rmat(quat)  # (K, B, 3, 3)
            
        elif self.rot_type == "dual_quat":
            transls = self.params["transls"][:, ts]  # (K, B, 3)
            quat = self.params["rots"][:, ts]  # (K, B, 4)
            dual_quat = quat_t_to_dq(quat, transls) # (G, K, 8)
            dual_quat = self.align_quat_to_max_coef(dual_quat, coefs)  # (G, K, B, 8)
            dual_quat = torch.einsum("pk,pkni->pni", coefs, dual_quat)
            quat, transls = dq_to_quat_t(dual_quat)
            rotmats = quat_to_rmat(quat)
            
        elif self.rot_type == "bingham": 
            transls = self.params["transls"][:, ts]  # (K, B, 3)
            transls = torch.einsum("pk,kni->pni", coefs, transls)
            quat = G_rots[:, ts]  # (G, B, 4)
            loss_dict['commit_loss'] = bingham_commit_loss(G_rots, coefs, self.params["rots"])
            rotmats = quat_to_rmat(quat)  # (K, B, 3, 3)
        
        if train:
            # KL divergence 
            coefs_mean = coefs.mean(dim=0)
            kl_transls = 0
            kl_rots = 0
            # mean
            prev_transls_mean = self.prev_transls[:, ts]
            prev_rots_mean = self.prev_rots[:, ts]
            curr_transls_mean = self.params.transls[:, ts]
            curr_rots_mean = self.params.rots[:, ts]
            # variance
            prev_rots_var = self.func_var_act(self.prev_rots_var[:, ts])
            prev_transls_var = self.func_var_act(self.prev_transls_var[:, ts])
            curr_rots_var = self.func_var_act(self.params["rots_var"][:, ts])
            curr_transls_var = self.func_var_act(self.params["transls_var"][:, ts])
            for i, coef in enumerate(coefs_mean):
                kl_transls += coef.detach() * self.func_kl_transls(prev_transls_mean[i], curr_transls_mean[i], 
                                                        prev_transls_var[i], curr_transls_var[i])
                kl_rots += coef.detach() * self.func_kl_transls(prev_rots_mean[i], curr_rots_mean[i], 
                                                        prev_rots_var[i], curr_rots_var[i])
            loss_dict['kl_transls'] = kl_transls 
            loss_dict['kl_rots'] = kl_rots
        return torch.cat([rotmats, transls[..., None]], dim=-1), loss_dict
    
    ###############################################################
    # aligning quaternion with coef
    ###############################################################
    def align_quat_to_max_coef(self, rots, coef):
        """
        Modify quaternions in 'rots' by comparing each quaternion with the one corresponding to the maximum coefficient 
        for each group in 'coef'. If the quaternions do not match, the sign of the quaternion is flipped.

        Args:
        rots (torch.Tensor): Quaternions of shape (K, ..., 4), where each quaternion is of the form (w, x, y, z).
        coef (torch.Tensor): Coefficients of shape (G, K) representing the coefficients for each quaternion.

        Returns:
        torch.Tensor: Modified quaternions with shape (G, K, ..., 4).
        """
        *leading_dims, _ = rots.shape
        G, B = coef.shape  # G is the number of groups, K is the basis num

        rots_expanded = rots.unsqueeze(0).expand(G, *leading_dims, -1)  # Shape becomes (G, K, ..., 4)

        max_coef_indices = coef.argmax(dim=-1)  # Shape is (G,)
        max_quats = rots_expanded[range(G), max_coef_indices]  # Shape becomes (G, ..., 4)

        signs = torch.sign(torch.sum(rots_expanded * max_quats.unsqueeze(1), dim=-1))  # Shape (G, K, ...)
        modified_quat = rots_expanded * signs.unsqueeze(-1)  # Shape becomes (G, K, ..., 4)
        return modified_quat

def check_gaussian_sizes(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    motion_coefs: torch.Tensor | None = None,
) -> bool:
    dims = means.shape[:-1]
    leading_dims_match = (
        quats.shape[:-1] == dims
        and scales.shape[:-1] == dims
        and colors.shape[:-1] == dims
        and opacities.shape == dims
    )
    if motion_coefs is not None and motion_coefs.numel() > 0:
        leading_dims_match &= motion_coefs.shape[:-1] == dims
    dims_correct = (
        means.shape[-1] == 3
        and (quats.shape[-1] == 4)
        and (scales.shape[-1] == 3)
        and (colors.shape[-1] == 3)
    )
    return leading_dims_match and dims_correct


def check_bases_sizes(motion_rots: torch.Tensor, motion_transls: torch.Tensor, rot_type: str) -> bool:
    if rot_type == '6d':
        rot_dim = 6
    elif rot_type in ['quat', 'dual_quat']:
        rot_dim = 4    
    elif rot_type == 'bingham':
        rot_dim = 8
    return (
        motion_rots.shape[-1] == rot_dim
        and motion_transls.shape[-1] == 3
        and motion_rots.shape[:-2] == motion_transls.shape[:-2]
    )
