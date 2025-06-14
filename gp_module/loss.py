import gpytorch

class WeightedVariationalELBO(gpytorch.mlls.VariationalELBO):
    def forward(self, variational_dist, target, confidence_mask=None):
        # 기본 ELBO 계산
        elbo = super().forward(variational_dist, target)
        
        if confidence_mask is not None:
            confidence_mask = confidence_mask.view_as(elbo)
            weighted_elbo = elbo * confidence_mask
            return weighted_elbo.sum() / confidence_mask.sum()
        
        return elbo

