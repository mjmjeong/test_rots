import gpytorch
import torch


class QuaternionKernel(gpytorch.kernels.Kernel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lengthscale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x1, x2, diag=False, **params):
        # Assume x1, x2 are quaternions (shape: [N, 4] or [N, T, 4])
        # Compute geodesic distance on S^3
        dot_product = (x1 * x2).sum(dim=-1)  # Cosine of angle
        angle = torch.acos(torch.clamp(dot_product, -1.0, 1.0))
        return torch.exp(-angle**2 / (2 * self.lengthscale**2))

class ProjectedGridMaternKernel(gpytorch.kernels.Kernel):
    def __init__(self, dims, grid_size=32, nu=1.5, grid_bounds=[[-1, 1]]*2, batch_shape=torch.Size([])):
        super().__init__(has_lengthscale=True)
        self.dims = dims  # tuple of two dimensions, e.g. (0, 1)

        base_matern = gpytorch.kernels.MaternKernel(nu=nu, ard_num_dims=2, batch_shape=batch_shape)
        print(base_matern.lengthscale)
        base_matern.lengthscale = 0.2
        print(base_matern.lengthscale)

        #base_matern.length_scale = 0.1
        base_matern.raw_lengthscale.requires_grad = True
        self.grid_kernel = gpytorch.kernels.GridInterpolationKernel(
            base_matern,
            grid_size=grid_size,
            num_dims=2,
            grid_bounds=grid_bounds,
        )
        

    def forward(self, x1, x2, **kwargs):
        return self.grid_kernel(x1[..., self.dims], x2[..., self.dims], **kwargs)

# Hexplane kernel
class Composite2DGridKernel(gpytorch.kernels.Kernel):
    def __init__(self, grid_size=[32, 32, 32, 32], grid_bounds=[(-1, 1),(-1, 1),(-1, 1),(-1, 1)], batch_shape=torch.Size([]), combine='add', nus=[2.5, 1.5]):
        super().__init__()
        nx, ny, nz, nt = grid_size
        
        # TODO: addictive / productive? Additive is more stable / productive is more complex but prune to overfitting
        if combine == 'add':
            self.kernels = gpytorch.kernels.AdditiveKernel(
                ProjectedGridMaternKernel(dims=(0, 1), grid_size=[nx, ny], nu=nus[0], grid_bounds=[grid_bounds[0], grid_bounds[1]], batch_shape=batch_shape),   # xy
                ProjectedGridMaternKernel(dims=(1, 2), grid_size=[ny, nz], nu=nus[0], grid_bounds=[grid_bounds[1], grid_bounds[2]], batch_shape=batch_shape),   # yz
                ProjectedGridMaternKernel(dims=(0, 2), grid_size=[nz, nx], nu=nus[0], grid_bounds=[grid_bounds[2], grid_bounds[0]], batch_shape=batch_shape),   # zx
                ProjectedGridMaternKernel(dims=(0, 3), grid_size=[nx, nt], nu=nus[1], grid_bounds=[grid_bounds[0], grid_bounds[3]], batch_shape=batch_shape),   # xt
                ProjectedGridMaternKernel(dims=(1, 3), grid_size=[ny, nt], nu=nus[1], grid_bounds=[grid_bounds[1], grid_bounds[3]], batch_shape=batch_shape),   # yt
                ProjectedGridMaternKernel(dims=(2, 3), grid_size=[nz, nt], nu=nus[1], grid_bounds=[grid_bounds[2], grid_bounds[3]], batch_shape=batch_shape),   # zt
            )
        else: 
            pass

    def forward(self, x1, x2, **kwargs):
        return self.kernels(x1, x2, **kwargs)



# Hexplane kernel
class Hexplane2DKernel(gpytorch.kernels.Kernel):
    def __init__(self, batch_shape=torch.Size([]), combine='add', nus=[2.5, 1.5]):
        super().__init__()
        nx, ny, nz, nt = grid_size
        
        # TODO: addictive / productive? Additive is more stable / productive is more complex but prune to overfitting
        if combine == 'add':
            self.kernels = gpytorch.kernels.AdditiveKernel(
                ProjectedGridMaternKernel(dims=(0, 1), grid_size=[nx, ny], nu=nus[0], grid_bounds=[grid_bounds[0], grid_bounds[1]], batch_shape=batch_shape),   # xy
                ProjectedGridMaternKernel(dims=(1, 2), grid_size=[ny, nz], nu=nus[0], grid_bounds=[grid_bounds[1], grid_bounds[2]], batch_shape=batch_shape),   # yz
                ProjectedGridMaternKernel(dims=(0, 2), grid_size=[nz, nx], nu=nus[0], grid_bounds=[grid_bounds[2], grid_bounds[0]], batch_shape=batch_shape),   # zx
                ProjectedGridMaternKernel(dims=(0, 3), grid_size=[nx, nt], nu=nus[1], grid_bounds=[grid_bounds[0], grid_bounds[3]], batch_shape=batch_shape),   # xt
                ProjectedGridMaternKernel(dims=(1, 3), grid_size=[ny, nt], nu=nus[1], grid_bounds=[grid_bounds[1], grid_bounds[3]], batch_shape=batch_shape),   # yt
                ProjectedGridMaternKernel(dims=(2, 3), grid_size=[nz, nt], nu=nus[1], grid_bounds=[grid_bounds[2], grid_bounds[3]], batch_shape=batch_shape),   # zt
            )
        else: 
            pass

    def forward(self, x1, x2, **kwargs):
        return self.kernels(x1, x2, **kwargs)


class SelectiveMaternKernel(gpytorch.kernels.Kernel):
    def __init__(self, dims, nu, batch_shape=torch.Size([])):
        super().__init__(has_lengthscale=True)
        self.dims = dims
        self.kernels = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=nu, ard_num_dims=len(self.dims), batch_shape=batch_shape),
                batch_shape=batch_shape)

    def forward(self, x1, x2, **kwargs):
        return self.kernels(x1[..., self.dims], x2[..., self.dims], **kwargs)

# Hexplane kernel
class HexplaneMaternKernel(gpytorch.kernels.Kernel):
    def __init__(self, batch_shape=torch.Size([]), combine='add', nus=[0.5, 0.5]):
        super().__init__()
        
        # TODO: addictive / productive? Additive is more stable / productive is more complex but prune to overfitting
        if combine == 'add':
            self.kernels = gpytorch.kernels.AdditiveKernel(
                SelectiveMaternKernel(dims=(0, 1), nu=nus[0], batch_shape=batch_shape),   # xy
                SelectiveMaternKernel(dims=(1, 2), nu=nus[0], batch_shape=batch_shape),   # yz
                SelectiveMaternKernel(dims=(2, 0), nu=nus[0], batch_shape=batch_shape),   # zx
                SelectiveMaternKernel(dims=(0, 3), nu=nus[1], batch_shape=batch_shape),   # xt
                SelectiveMaternKernel(dims=(1, 3), nu=nus[1], batch_shape=batch_shape),   # yt
                SelectiveMaternKernel(dims=(2, 3), nu=nus[1], batch_shape=batch_shape),   # zt
            )
        elif combine == 'prod':
            self.kernels = gpytorch.kernels.ProductKernel(
                SelectiveMaternKernel(dims=(0, 1), nu=nus[0], batch_shape=batch_shape),   # xy
                SelectiveMaternKernel(dims=(1, 2), nu=nus[0], batch_shape=batch_shape),   # yz
                SelectiveMaternKernel(dims=(2, 0), nu=nus[0], batch_shape=batch_shape),   # zx
                SelectiveMaternKernel(dims=(0, 3), nu=nus[1], batch_shape=batch_shape),   # xt
                SelectiveMaternKernel(dims=(1, 3), nu=nus[1], batch_shape=batch_shape),   # yt
                SelectiveMaternKernel(dims=(2, 3), nu=nus[1], batch_shape=batch_shape),   # zt
            )

    def forward(self, x1, x2, **kwargs):
        return self.kernels(x1, x2, **kwargs)

