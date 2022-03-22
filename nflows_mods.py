import torch
from torch.nn.functional import softplus
from nflows import transforms, nn

# AffineCouplingTransform modified to allow scales >1.001
class AffineCouplingTransform(transforms.AffineCouplingTransform):
    DEFAULT_SCALE_ACTIVATION = lambda x : torch.sigmoid(x + 2) + 1e-3
    GENERAL_SCALE_ACTIVATION = lambda x : (softplus(x) + 1e-3).clamp(0, 3)

    def __init__(self, mask, transform_net_create_fn, unconditional_transform=None,
                 scale_activation=DEFAULT_SCALE_ACTIVATION):
        self.scale_activation = scale_activation
        super().__init__(mask, transform_net_create_fn, unconditional_transform)

    def _scale_and_shift(self, transform_params):
        unconstrained_scale = transform_params[:, self.num_transform_features:, ...]
        shift = transform_params[:, : self.num_transform_features, ...]
        scale = self.scale_activation(unconstrained_scale)
        return scale, shift

# MLP modified to add a context argument to forward (but using this argument is unimplemented)
class MLP(nn.nets.MLP):
    def forward(self, inputs, context=None):
        if context is not None:
            raise NotImplementedError()
        return super().forward(inputs)
