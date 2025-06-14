import gpytorch

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
