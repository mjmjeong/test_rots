from typing import Literal

import roma
import torch
import torch.nn.functional as F


def rt_to_mat4(
    R: torch.Tensor, t: torch.Tensor, s: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Args:
        R (torch.Tensor): (..., 3, 3).
        t (torch.Tensor): (..., 3).
        s (torch.Tensor): (...,).

    Returns:
        torch.Tensor: (..., 4, 4)
    """
    mat34 = torch.cat([R, t[..., None]], dim=-1)
    if s is None:
        bottom = (
            mat34.new_tensor([[0.0, 0.0, 0.0, 1.0]])
            .reshape((1,) * (mat34.dim() - 2) + (1, 4))
            .expand(mat34.shape[:-2] + (1, 4))
        )
    else:
        bottom = F.pad(1.0 / s[..., None, None], (3, 0), value=0.0)
    mat4 = torch.cat([mat34, bottom], dim=-2)
    return mat4


def rmat_to_cont_6d(matrix):
    """
    :param matrix (*, 3, 3)
    :returns 6d vector (*, 6)
    """
    return torch.cat([matrix[..., 0], matrix[..., 1]], dim=-1)


def cont_6d_to_rmat(cont_6d):
    """
    :param 6d vector (*, 6)
    :returns matrix (*, 3, 3)
    """
    x1 = cont_6d[..., 0:3]
    y1 = cont_6d[..., 3:6]

    x = F.normalize(x1, dim=-1)
    y = F.normalize(y1 - (y1 * x).sum(dim=-1, keepdim=True) * x, dim=-1)
    z = torch.linalg.cross(x, y, dim=-1)

    return torch.stack([x, y, z], dim=-1)

def quat_to_rmat(quat):
    """
    Converts a quaternion (w, x, y, z) to a rotation matrix (3x3) using torch.
    The input quaternion is of shape (..., 4), where N is the number of quaternions.
    The output is a rotation matrix of shape (..., 3, 3).
    """
    # Normalize the quaternions (optional)
    quat = quat / quat.norm(dim=-1, keepdim=True)
    *leading_dims, _ = quat.shape
    w = quat[..., 0]
    x = quat[..., 1]
    y = quat[..., 2]
    z = quat[..., 3]
    
    # Compute the rotation matrix from quaternion
    rot_matrix = torch.zeros((*leading_dims, 3, 3), device=quat.device)
    
    rot_matrix[..., 0, 0] = 1 - 2 * (y ** 2 + z ** 2)
    rot_matrix[..., 0, 1] = 2 * (x * y - z * w)
    rot_matrix[..., 0, 2] = 2 * (x * z + y * w)
    
    rot_matrix[..., 1, 0] = 2 * (x * y + z * w)
    rot_matrix[..., 1, 1] = 1 - 2 * (x ** 2 + z ** 2)
    rot_matrix[..., 1, 2] = 2 * (y * z - x * w)
    
    rot_matrix[..., 2, 0] = 2 * (x * z - y * w)
    rot_matrix[..., 2, 1] = 2 * (y * z + x * w)
    rot_matrix[..., 2, 2] = 1 - 2 * (x ** 2 + y ** 2)
    
    return rot_matrix

def quat_multiply(q1, q2):
    """
    Multiply two quaternions q1 and q2 (batch-wise).
    :param q1: First quaternion (batch_size, 4)
    :param q2: Second quaternion (batch_size, 4)
    :returns: Resulting quaternion (batch_size, 4)
    """
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2

    return torch.stack((w, x, y, z), dim=-1)

def quat_t_to_dq(q_r, T):
    """
    Converts a rotation quaternion q_r and a translation vector T to a dual quaternion.
    :param q_r: Rotation quaternion of shape (..., 4)
    :param T: Translation vector of shape (..., 3)
    :returns: Dual quaternion of shape (..., 8)
    """
    # Normalize the rotation quaternion (optional, ensure it's unit quaternion)
    q_r = q_r / q_r.norm(dim=-1, keepdim=True)

    # Create the translation quaternion (0, T_x, T_y, T_z)
    q_t = torch.zeros_like(q_r)
    q_t[..., 1:] = T  # Set the translation part
    
    # Compute the dual quaternion: q = q_r + 1/2 * q_r * q_t
    q_t = 0.5 * (quat_multiply(q_r, q_t))  # quaternion multiplication
    #q_t[..., 0] = 0 #TODO: check the ablation
    dual_quat = torch.cat((q_r, q_t), dim=-1)
    return dual_quat


def dq_to_quat_t(dq):
    """
    dq is a tensor of shape (..., 8), where each dual quaternion is represented by 8 values (w_r, x_r, y_r, z_r, w_t, x_t, y_t, z_t).
    Returns qauternion and translation (..., 4) and (..., 3).
    """
    # Extract rotation quaternion (q_r) and translation quaternion (q_t)
    q_r = dq[..., :4]  # First 4 values for rotation
    q_t = dq[..., 4:]  # Last 4 values for translation

    # Normalize the rotation quaternion (optional)
    q_r = q_r / q_r.norm(dim=-1, keepdim=True)

    # Compute translation vector T from the dual quaternion q_r and q_t
    q_r_conjugate = torch.cat([q_r[..., 0:1], -q_r[..., 1:]], dim=-1)  # Conjugate of the rotation quaternion

    # Compute the translation part using quaternion multiplication
    T = 2 * quat_multiply(q_r_conjugate, q_t)[..., 1:]
    return q_r, T

def solve_procrustes(
    src: torch.Tensor,
    dst: torch.Tensor,
    weights: torch.Tensor | None = None,
    enforce_se3: bool = False,
    rot_type: Literal["quat", "mat", "6d", 'dual_quat'] = "quat",
):
    """
    Solve the Procrustes problem to align two point clouds, by solving the
    following problem:

    min_{s, R, t} || s * (src @ R.T + t) - dst ||_2, s.t. R.T @ R = I and det(R) = 1.

    Args:
        src (torch.Tensor): (N, 3).
        dst (torch.Tensor): (N, 3).
        weights (torch.Tensor | None): (N,), optional weights for alignment.
        enforce_se3 (bool): Whether to enforce the transfm to be SE3.

    Returns:
        sim3 (tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
            q (torch.Tensor): (4,), rotation component in quaternion of WXYZ
                format.
            t (torch.Tensor): (3,), translation component.
            s (torch.Tensor): (), scale component.
        error (torch.Tensor): (), average L2 distance after alignment.
    """
    # Compute weights.
    if weights is None:
        weights = src.new_ones(src.shape[0])
    weights = weights[:, None] / weights.sum()
    # Normalize point positions.
    src_mean = (src * weights).sum(dim=0)
    dst_mean = (dst * weights).sum(dim=0)
    src_cent = src - src_mean
    dst_cent = dst - dst_mean
    # Normalize point scales.
    if not enforce_se3:
        src_scale = (src_cent**2 * weights).sum(dim=-1).mean().sqrt()
        dst_scale = (dst_cent**2 * weights).sum(dim=-1).mean().sqrt()
    else:
        src_scale = dst_scale = src.new_tensor(1.0)
    src_scaled = src_cent / src_scale
    dst_scaled = dst_cent / dst_scale
    # Compute the matrix for the singular value decomposition (SVD).
    matrix = (weights * dst_scaled).T @ src_scaled
    U, _, Vh = torch.linalg.svd(matrix)
    # Special reflection case.
    S = torch.eye(3, device=src.device)
    if torch.det(U) * torch.det(Vh) < 0:
        S[2, 2] = -1
    R = U @ S @ Vh
    # Compute the transformation.
    if rot_type in ["quat", "dual_quat"]:
        rot = roma.rotmat_to_unitquat(R).roll(1, dims=-1)
    elif rot_type == "6d":
        rot = rmat_to_cont_6d(R)
    else:
        rot = R
    s = dst_scale / src_scale
    t = dst_mean / s - src_mean @ R.T
    sim3 = rot, t, s
    # Debug: error.
    procrustes_dst = torch.einsum(
        "ij,nj->ni", rt_to_mat4(R, t, s), F.pad(src, (0, 1), value=1.0)
    )
    procrustes_dst = procrustes_dst[:, :3] / procrustes_dst[:, 3:]
    error_before = (torch.linalg.norm(dst - src, dim=-1) * weights[:, 0]).sum()
    error = (torch.linalg.norm(dst - procrustes_dst, dim=-1) * weights[:, 0]).sum()
    # print(f"Procrustes error: {error_before} -> {error}")
    # if error_before < error:
    #     print("Something is wrong.")
    #     __import__("ipdb").set_trace()
    return sim3, (error.item(), error_before.item())
