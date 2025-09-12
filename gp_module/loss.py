import gpytorch
import torch.nn.functional as F

class WeightedVariationalELBO(gpytorch.mlls.VariationalELBO):
    def forward(self, variational_dist, target, confidence_mask=None):
        # 기본 ELBO 계산
        elbo = super().forward(variational_dist, target)
        
        if confidence_mask is not None:
            confidence_mask = confidence_mask.view_as(elbo)
            weighted_elbo = elbo * confidence_mask
            return weighted_elbo.sum() / confidence_mask.sum()
        
        return elbo

def gp_gs_loss(cfg, pred, gs_mean, gs_val):
    if cfg.gp_gs_loss_type == 'l1':
        loss = F.l1_loss(pred, gs_mean)
    elif cfg.gp_gs_loss_type == 'mse':
        loss = F.mse_loss(pred, gs_mean)
    elif cfg.gp_gs_loss_type == 'var_l1':
        l1_loss = F.l1_loss(pred, gs_mean, reduction='none')
        gs_val = gs_val ** (1/cfg.variance_scaling)
        loss = (l1_loss * gs_val).mean()
    elif cfg.gp_gs_loss_type == 'var_mse':
        mse_loss = F.mse_loss(pred, gs_mean, reduction='none')
        gs_val = gs_val ** (1/cfg.variance_scaling)
        loss = (mse_loss * gs_val).mean()
    return loss