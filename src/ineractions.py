import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from relu import ReLUKANLayer
from kan import KANLayer


class CustomReLUKANLayer(ReLUKANLayer):
    def __init__(self, in_features: int, out_features: int, base_activation: str, grid_size: int, k: int, train_boundaries: bool = True):
        super(CustomReLUKANLayer, self).__init__(in_features, out_features, base_activation, grid_size, k, train_boundaries)



    def forward(self, x: torch.Tensor):
            x_base = super().forward(x)
            interaction = x_base @ x_base.T
            interaction_summed = interaction.sum(dim=-1)
            return x_base + interaction_summed.unsqueeze(1) 
    



#TODO apply decorators for interactions, do not repeat classes

# Decorator to apply custom interaction
def apply_custom_interaction(forward_func):
    @functools.wraps(forward_func)
    def wrapper(self, x: torch.Tensor):
        if self.apply_interaction:  
            # First forward pass through the base layer
            x_base = forward_func(self, x)
            
            interaction = x_base @ x_base.T
            interaction_summed = interaction.sum(dim=-1)
            return x_base + interaction_summed.unsqueeze(1)
        else:
            # If interaction is not applied, return the base forward result
            return forward_func(self, x)
    
    return wrapper


class CustomKANLayer(KANLayer):
    def __init__(self,      
        in_features,
        out_features,
        base_activation="silu",
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        grid_eps=0.02,
        grid_range=[-1, 1]):
        super(CustomKANLayer, self).__init__(
            in_features,
            out_features,
            base_activation,
            grid_size,
            spline_order,
            scale_noise,
            scale_base,
            scale_spline,
            enable_standalone_scale_spline,
            grid_eps,
            grid_range)



    def forward(self, x: torch.Tensor, update_grid=False):
        # Call the parent class forward method
            x_base = super(CustomKANLayer, self).forward(x, update_grid)
            # print(f"x_base shape: {x_base.shape}")               # Expected: (batch_size, out_features)

            # Compute interaction term: x @ x.T
            interaction = x_base @ x_base.T

            # print(f"interaction shape: {interaction.shape}")  # Expected: (batch_size, out_features)


            # Scale the interaction term by learnable alpha parameters (element-wise multiplication)
            # interaction_scaled = self.alpha * interaction

            # Sum along the last dimension (as per your requirement)
            # interaction_summed = interaction_scaled.sum(dim=-1)

            interaction_summed = interaction.sum(dim=-1)

            # print(f"interaction_summed shape: {interaction_summed.shape}")  # Expected: (batch_size, out_features)

            #why unsqueeze?
            #because x_base shape: torch.Size([64, 10])
            # interaction shape: torch.Size([64, 64])
            # interaction_summed shape: torch.Size([64])

            return x_base + interaction_summed.unsqueeze(1) 
           