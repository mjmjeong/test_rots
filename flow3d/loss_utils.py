import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors


def masked_mse_loss(pred, gt, mask=None, normalize=True, quantile: float = 1.0):
    if mask is None:
        return trimmed_mse_loss(pred, gt, quantile)
    else:
        sum_loss = F.mse_loss(pred, gt, reduction="none").mean(dim=-1, keepdim=True)
        quantile_mask = (
            (sum_loss < torch.quantile(sum_loss, quantile)).squeeze(-1)
            if quantile < 1
            else torch.ones_like(sum_loss, dtype=torch.bool).squeeze(-1)
        )
        ndim = sum_loss.shape[-1]
        if normalize:
            return torch.sum((sum_loss * mask)[quantile_mask]) / (
                ndim * torch.sum(mask[quantile_mask]) + 1e-8
            )
        else:
            return torch.mean((sum_loss * mask)[quantile_mask])


# def masked_l1_loss(pred, gt, mask=None, normalize=True, quantile: float = 1.0):
#     if mask is None:
#         return trimmed_l1_loss(pred, gt, quantile)
#     else:
#         sum_loss = F.l1_loss(pred, gt, reduction="none").mean(dim=-1, keepdim=True)
#         quantile_mask = (
#             (sum_loss < torch.quantile(sum_loss, quantile)).squeeze(-1)
#             if quantile < 1
#             else torch.ones_like(sum_loss, dtype=torch.bool).squeeze(-1)
#         )
#         ndim = sum_loss.shape[-1]
#         if normalize:
#             return torch.sum((sum_loss * mask)[quantile_mask]) / (
#                 ndim * torch.sum(mask[quantile_mask]) + 1e-8
#             )
#         else:
#             return torch.mean((sum_loss * mask)[quantile_mask])


def masked_l1_loss(pred, gt, mask=None, normalize=True, quantile: float = 1.0):
    if mask is None:
        return trimmed_l1_loss(pred, gt, quantile)
    else:
        sum_loss = F.l1_loss(pred, gt, reduction="none").mean(dim=-1, keepdim=True)
        # sum_loss.shape 
        # block     [218255, 1]
        # apple     [36673, 475, 1]     17,419,675
        # creeper   [37587, 360, 1]     13,531,320
        # backpack  [37828, 180, 1]     6,809,040
        # quantile_mask = (
        #     (sum_loss < torch.quantile(sum_loss, quantile)).squeeze(-1)
        #     if quantile < 1
        #     else torch.ones_like(sum_loss, dtype=torch.bool).squeeze(-1)
        # )
        # use torch.sort instead of torch.quantile when input too large
        if quantile < 1:
            num = sum_loss.numel()
            if num < 16_000_000:
                threshold = torch.quantile(sum_loss, quantile)
            else:
                sorted, _ = torch.sort(sum_loss.reshape(-1))
                idxf = quantile * num
                idxi = int(idxf)
                threshold = sorted[idxi] + (sorted[idxi + 1] - sorted[idxi]) * (idxf - idxi)
            quantile_mask = (sum_loss < threshold).squeeze(-1)
        else: 
            quantile_mask = torch.ones_like(sum_loss, dtype=torch.bool).squeeze(-1)

        ndim = sum_loss.shape[-1]
        if normalize:
            return torch.sum((sum_loss * mask)[quantile_mask]) / (
                ndim * torch.sum(mask[quantile_mask]) + 1e-8
            )
        else:
            return torch.mean((sum_loss * mask)[quantile_mask])

def masked_huber_loss(pred, gt, delta, mask=None, normalize=True):
    if mask is None:
        return F.huber_loss(pred, gt, delta=delta)
    else:
        sum_loss = F.huber_loss(pred, gt, delta=delta, reduction="none")
        ndim = sum_loss.shape[-1]
        if normalize:
            return torch.sum(sum_loss * mask) / (ndim * torch.sum(mask) + 1e-8)
        else:
            return torch.mean(sum_loss * mask)


def trimmed_mse_loss(pred, gt, quantile=0.9):
    loss = F.mse_loss(pred, gt, reduction="none").mean(dim=-1)
    loss_at_quantile = torch.quantile(loss, quantile)
    trimmed_loss = loss[loss < loss_at_quantile].mean()
    return trimmed_loss


def trimmed_l1_loss(pred, gt, quantile=0.9):
    loss = F.l1_loss(pred, gt, reduction="none").mean(dim=-1)
    loss_at_quantile = torch.quantile(loss, quantile)
    trimmed_loss = loss[loss < loss_at_quantile].mean()
    return trimmed_loss


def compute_gradient_loss(pred, gt, mask, quantile=0.98):
    """
    Compute gradient loss
    pred: (batch_size, H, W, D) or (batch_size, H, W)
    gt: (batch_size, H, W, D) or (batch_size, H, W)
    mask: (batch_size, H, W), bool or float
    """
    # NOTE: messy need to be cleaned up
    mask_x = mask[:, :, 1:] * mask[:, :, :-1]
    mask_y = mask[:, 1:, :] * mask[:, :-1, :]
    pred_grad_x = pred[:, :, 1:] - pred[:, :, :-1]
    pred_grad_y = pred[:, 1:, :] - pred[:, :-1, :]
    gt_grad_x = gt[:, :, 1:] - gt[:, :, :-1]
    gt_grad_y = gt[:, 1:, :] - gt[:, :-1, :]
    loss = masked_l1_loss(
        pred_grad_x[mask_x][..., None], gt_grad_x[mask_x][..., None], quantile=quantile
    ) + masked_l1_loss(
        pred_grad_y[mask_y][..., None], gt_grad_y[mask_y][..., None], quantile=quantile
    )
    return loss


def knn(x: torch.Tensor, k: int) -> tuple[np.ndarray, np.ndarray]:
    x = x.cpu().numpy()
    knn_model = NearestNeighbors(
        n_neighbors=k + 1, algorithm="auto", metric="euclidean"
    ).fit(x)
    distances, indices = knn_model.kneighbors(x)
    return distances[:, 1:].astype(np.float32), indices[:, 1:].astype(np.float32)


def get_weights_for_procrustes(clusters, visibilities=None):
    clusters_median = clusters.median(dim=-2, keepdim=True)[0]
    dists2clusters_center = torch.norm(clusters - clusters_median, dim=-1)
    dists2clusters_center /= dists2clusters_center.median(dim=-1, keepdim=True)[0]
    weights = torch.exp(-dists2clusters_center)
    weights /= weights.mean(dim=-1, keepdim=True) + 1e-6
    if visibilities is not None:
        weights *= visibilities.float() + 1e-6
    invalid = dists2clusters_center > np.quantile(
        dists2clusters_center.cpu().numpy(), 0.9
    )
    invalid |= torch.isnan(weights)
    weights[invalid] = 0
    return weights


def compute_z_acc_loss(means_ts_nb: torch.Tensor, w2cs: torch.Tensor):
    """
    :param means_ts (G, 3, B, 3)
    :param w2cs (B, 4, 4)
    return (float)
    """
    camera_center_t = torch.linalg.inv(w2cs)[:, :3, 3]  # (B, 3)
    ray_dir = F.normalize(
        means_ts_nb[:, 1] - camera_center_t, p=2.0, dim=-1
    )  # [G, B, 3]
    # acc = 2 * means[:, 1] - means[:, 0] - means[:, 2]  # [G, B, 3]
    # acc_loss = (acc * ray_dir).sum(dim=-1).abs().mean()
    acc_loss = (
        ((means_ts_nb[:, 1] - means_ts_nb[:, 0]) * ray_dir).sum(dim=-1) ** 2
    ).mean() + (
        ((means_ts_nb[:, 2] - means_ts_nb[:, 1]) * ray_dir).sum(dim=-1) ** 2
    ).mean()
    return acc_loss


def compute_se3_smoothness_loss(
    rots: torch.Tensor,
    transls: torch.Tensor,
    weight_rot: float = 1.0,
    weight_transl: float = 2.0,
):
    """
    central differences
    :param motion_transls (K, T, 3)
    :param motion_rots (K, T, 6)
    """
    r_accel_loss = compute_accel_loss(rots)
    t_accel_loss = compute_accel_loss(transls)
    return r_accel_loss * weight_rot + t_accel_loss * weight_transl


def compute_accel_loss(transls):
    accel = 2 * transls[:, 1:-1] - transls[:, :-2] - transls[:, 2:]
    loss = accel.norm(dim=-1).mean()
    return loss

def kl_div_gaussian(pre_mean, curr_mean, pre_var, curr_var):
    """
    Compute the KL divergence between two Gaussian distributions.
    
    Args:
        pre_mean (torch.Tensor): Mean of the prior distribution, shape (B, N).
        curr_mean (torch.Tensor): Mean of the approximate posterior distribution, shape (B, N).
        pre_var (torch.Tensor): Variance of the prior distribution, shape (B, N).
        curr_var (torch.Tensor): Variance of the approximate posterior distribution, shape (B, N).
    
    Returns:
        torch.Tensor: The KL divergence between the two distributions, shape (B,).
    """
    # Compute the KL divergence between two Gaussians (element-wise)
    log_var_ratio = torch.log(curr_var / pre_var)  # log(σ2^2 / σ1^2)
    var_ratio = curr_var / pre_var  #σ1^2 / σ2^2)

    # Mean difference squared term: (μ2 - μ1)^2 / (2 * σ1^2)
    mean_diff = (curr_mean - pre_mean)**2
    mean_term = mean_diff / (2 * pre_var)
    
    # KL divergence formula (element-wise)
    kl_div = 0.5 * (log_var_ratio + var_ratio + mean_term - 1)
    return kl_div.mean()

def kl_div_vonmise_fisher():
    pass

def reparameterize_vonmise_fisher(vmf_kappa, axis):
    # sampling visualization
    pass

def bingham_loglikelihood(vec10, quat, normalization=False):
    """
    Compute the log-likelihood of the Bingham distribution for quaternions using element-wise operations.
    
    Parameters:
    - vec10: torch.Tensor of shape (N, 10) containing the Bingham parameters
            format: [a, b, c, d, e, f, g, h, i, j] which represent elements of the 4x4 symmetric matrix A
    - quat: torch.Tensor of shape (N, 4) containing quaternions
    
    Returns:
    - log_likelihood: torch.Tensor of shape (N,) containing the log-likelihood for each quaternion
    """
    N = quat.shape[0]
    
    # Normalize the quaternions
    quat_norm = torch.nn.functional.normalize(quat, p=2, dim=1)
    
    # Extract quaternion components
    q0 = quat_norm[:, 0]
    q1 = quat_norm[:, 1]
    q2 = quat_norm[:, 2]
    q3 = quat_norm[:, 3]
    
    # Extract Bingham matrix parameters
    a = vec10[:, 0]  # A[0,0]
    b = vec10[:, 1]  # A[1,1]
    c = vec10[:, 2]  # A[2,2]
    d = vec10[:, 3]  # A[3,3]
    e = vec10[:, 4]  # A[0,1] = A[1,0]
    f = vec10[:, 5]  # A[0,2] = A[2,0]
    g = vec10[:, 6]  # A[0,3] = A[3,0]
    h = vec10[:, 7]  # A[1,2] = A[2,1]
    i = vec10[:, 8]  # A[1,3] = A[3,1]
    j = vec10[:, 9]  # A[2,3] = A[3,2]
    
    # Compute the diagonal terms
    diag_terms = a * q0 * q0 + b * q1 * q1 + c * q2 * q2 + d * q3 * q3
    
    # Compute the off-diagonal terms (each appears twice in the quadratic form)
    off_diag_terms = 2 * (e * q0 * q1 + 
                        f * q0 * q2 + 
                        g * q0 * q3 + 
                        h * q1 * q2 + 
                        i * q1 * q3 + 
                        j * q2 * q3)
    
    # Compute the quadratic form q^T·A·q
    quadratic_form = diag_terms + off_diag_terms
    
    # Compute log normalization constant
    # This is a simplified approach and may need to be adjusted based on specific requirements
    if normalization:
        log_likelihood =  -1 * (quadratic_form - log_F)
    else:
        log_likelihood =  -1 * quadratic_form
    
    return log_likelihood

def bingham_loglikelihood_matmul(vec10, quat, normalization=False):
    """
    Compute the log-likelihood of the Bingham distribution using matrix multiplication.
    This function is provided for verification purposes.
    """
    N = quat.shape[0]
    
    # Normalize the quaternions
    quat_norm = torch.nn.functional.normalize(quat, p=2, dim=1)
    
    # Initialize batch of 4x4 symmetric matrices
    A = torch.zeros(N, 4, 4, device=quat.device)
    
    # Fill in the diagonal elements
    A[:, 0, 0] = vec10[:, 0]  # a
    A[:, 1, 1] = vec10[:, 1]  # b
    A[:, 2, 2] = vec10[:, 2]  # c
    A[:, 3, 3] = vec10[:, 3]  # d
    
    # Fill in the off-diagonal elements
    A[:, 0, 1] = A[:, 1, 0] = vec10[:, 4]  # e
    A[:, 0, 2] = A[:, 2, 0] = vec10[:, 5]  # f
    A[:, 0, 3] = A[:, 3, 0] = vec10[:, 6]  # g
    A[:, 1, 2] = A[:, 2, 1] = vec10[:, 7]  # h
    A[:, 1, 3] = A[:, 3, 1] = vec10[:, 8]  # i
    A[:, 2, 3] = A[:, 3, 2] = vec10[:, 9]  # j
    
    # Reshape quat for batch matrix multiplication
    q = quat_norm.unsqueeze(2)  # Shape: (N, 4, 1)
    
    # Compute q^T·A·q using batch matrix multiplication
    # First compute A·q: (N, 4, 4) × (N, 4, 1) = (N, 4, 1)
    Aq = torch.bmm(A, q)
    
    # Then compute q^T·(A·q): (N, 1, 4) × (N, 4, 1) = (N, 1, 1)
    qAq = torch.bmm(q.transpose(1, 2), Aq)
    
    # Compute the quadratic form q^T·A·q
    quadratic_form = qAq.squeeze()
    
    # Compute log normalization constant
    log_F = torch.zeros(N, device=vec10.device)
    
    if normalization:
        log_likelihood =  -1 * (quadratic_form - log_F)
    else:
        log_likelihood =  -1 * quadratic_form
    
    return log_likelihood


def compute_bingham_geometry(vec10, return_entropy=False, return_concent=False):
    # 10 vec -> mat
    matrices = bingham_recon_mta(vec10)  # (N, 4, 4)
    eigenvalues, eigenvectors = torch.linalg.eigh(matrices)
    
    concentration_neg_min, _ = -eigenvalues.min(-1) # Negative of smallest eigenvalue
    concentration_sum_neg = -torch.sum(eigenvalues, dim=1)
    
    if return_entropy:
        pass
    concentration_var = torch.var(eigenvalues, dim=1)
    # Return all concentration measures for analysis
    return {
        'eigenvectors': eigenvectors
    }