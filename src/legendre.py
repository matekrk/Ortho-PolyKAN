import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import lru_cache
from polynomial import PolynomialLayer


class LegendreLayer(PolynomialLayer):
    def __init__(self, in_features: int, out_features: int, base_activation: str, polynomial_order: int, layer_norm: bool = True):
        super().__init__(in_features, out_features, base_activation, polynomial_order, layer_norm)

    @lru_cache(maxsize=128)  # Cache to avoid recomputation of Legendre polynomials
    def compute_polynomials(self, x: torch.Tensor, order: int):
        # Base case polynomials P0 and P1
        P0 = x.new_ones(x.shape)  # P0 = 1 for all x
        if order == 0:
            return P0.unsqueeze(-1)
        P1 = x  # P1 = x
        legendre_polys = [P0, P1]
        
        # Compute higher order polynomials using recurrence
        for n in range(1, order):
            Pn = ((2.0 * n + 1.0) * x * legendre_polys[-1] - n * legendre_polys[-2]) / (n + 1.0)
            legendre_polys.append(Pn)
        
        return torch.stack(legendre_polys, dim=-1)
