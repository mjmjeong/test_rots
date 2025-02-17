import gsplat
import torch
from torch.nn import functional as F
import mediapy as media

num_gaussians = 1000
means = F.normalize(torch.randn(num_gaussians, 3, device="cuda"), dim=-1)
means[..., 2] = torch.rand(num_gaussians) * 3 + 2
scales = torch.rand_like(means) * 0.1 + 0.05
quaternions = torch.tensor([1.0, 0.0, 0.0, 0.0], device="cuda")[None, :].expand(
    num_gaussians, 4
)
opacities = torch.rand_like(scales[..., 0])
rgbs = torch.rand_like(means)

num_cameras = 4
viewmats = torch.eye(4, device="cuda")[None].expand(num_cameras, 4, 4)
Ks = torch.tensor(
    [
        [300.0, 0.0, 150.0],
        [0.0, 300.0, 100.0],
        [0.0, 0.0, 1.0],
    ],
    device="cuda",
)[None, :, :].expand(num_cameras, 3, 3)
width, height = 300, 200

features, alphas, normals, surf_normals, dist_loss, depth_median, _ = gsplat.rasterization_2dgs(
    means=means,
    quats=quaternions,
    opacities=opacities,
    colors=rgbs,
    scales=scales,
    viewmats=viewmats,
    Ks=Ks,
    width=width,
    height=height,
    render_mode="RGB+D",
)
media.show_images(features[..., :3].numpy(force=True))
media.show_images(features[..., 3:].numpy(force=True), vmin=0, vmax=5)
media.show_images(depth_median.numpy(force=True), vmin=0, vmax=5)
media.show_images(normals.numpy(force=True)*0.5+0.5)
media.show_images(surf_normals.numpy(force=True)*0.5+0.5)
