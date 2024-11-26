import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from relu import ReLUKANLayer,ReLUKANNetwork, Union, List
from kan import KANLayer


import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomReLUKANLayer(ReLUKANLayer):
    def __init__(self, in_features: int, out_features: int, base_activation: str, grid_size: int, k: int, train_boundaries: bool = True,apply_interaction=True):
        super(CustomReLUKANLayer, self).__init__(in_features, out_features, base_activation, grid_size, k, train_boundaries)
        
        # phase_low = torch.arange(-self.k, self.grid_size) / self.grid_size
        # phase_height = phase_low + (self.k+1) / self.grid_size
        # self.phase_low = nn.Parameter(phase_low[None, :].expand(in_features, -1).clone(), requires_grad=train_boundaries)
        # self.phase_height = nn.Parameter(phase_height[None, :].expand(in_features, -1).clone(), requires_grad=train_boundaries)
        
  
        self.alpha = nn.Parameter(torch.randn(out_features, out_features))  
        self.apply_interaction = apply_interaction

        # Tworzymy "sztuczną konwolucję" z wagami 1
        self.equal_size_conv_no_params = nn.Conv2d(self.hidden_dim, out_features, (self.grid_size+self.k, in_features))

        self.equal_size_conv_no_params.weight.data.fill_(1)  # Ustawiamy wszystkie wagi na 1
        self.equal_size_conv_no_params.bias.data.zero_()     # Ustawiamy bias na 0 (jeśli istnieje)
        self.equal_size_conv_no_params.weight.requires_grad = False  # Zamrażamy wagi
        self.equal_size_conv_no_params.bias.requires_grad = False    # Zamrażamy bias

    def forward(self, x: torch.Tensor, apply_interactions: bool = False):
        print(f"infeature : {self.in_features}") 
        print(f"outfeature : {self.out_features}") 
        print(f"k : {self.k}") 
        print(f"x : {x.shape}") 
        
        # Original basis function computation (with change in Eq 11???)
        x_expanded = x.unsqueeze(2).expand(-1, -1, self.phase_low.size(1))
        print(f"x_expanded : {x_expanded.shape}") 
        #Eq 9
        x1 = self.base_activation(x_expanded - self.phase_low)
        #Eq 10
        x2 = self.base_activation(self.phase_height - x_expanded)
        #Eq 11
        x = x1 * x2 * self.r 
        # Eq 12
        x = x * x  
        print(f"F : {x.shape}") #check out 1568 (concatenated?)
        # Eq 13
        x = x.reshape((len(x), 1, self.grid_size + self.k, self.in_features))
        print(f"reshaped x : {x.shape}") 
        phi = self.equal_size_conv(x) #consider changing conv to Eq 13 directly, przemnóż każdą zmienną * każdą chcemy aby miało 64,10
        print(f"phi : {phi.shape}") 
        phi = phi.reshape((len(phi), self.out_features))
        print(f"reshaped phi : {phi.shape}") 

          ### Added ###
        if self.apply_interaction:
       
                # takie samo jak phi ale ze wspolczynnikami konwolucji = 1
                B = self.equal_size_conv_no_params(x)
                B = B.reshape((len(B), self.out_features))
                

            
                # Mnożenie każdej zmiennej przez każdą zmienną
                interaction = torch.einsum('bi,bj->bij', phi, phi)  # Rozmiar: batch_size x out_features x out_features
                
                

                # Używa wyrażenia bi,bj->bij, które oznacza: dla każdej próbki w batchu (b), weź każdą funkcję wyjściową 
                # i przemnóż ją przez każdą funkcję wyjściową j.
                
                #Powinno działać tak samo:
                phi_outer = torch.bmm(phi.unsqueeze(-1), phi.unsqueeze(1))

                # print(phi_outer==interaction)

                print(f" interaction shape : {interaction.shape}") 

                weighted_interaction =  interaction * self.alpha    # Element-wise multiplication

                print(f" weighted interaction  : {weighted_interaction.shape}") 

                # Jeśli wagi mają wpływać na sumaryczne interakcje między zmiennymi, używamy sumowania przed mnożeniem.
                # Jeśli wagi mają wpływać na każdą indywidualną interakcję, używamy sumowania po mnożeniu.
                # Tu wybrałam drugie podejście
                weighted_interaction = weighted_interaction.sum(dim=-1)  # Rozmiar: (batch_size, out_features)
                print(f" weighted interaction summed : {weighted_interaction.shape}") 
        
                return phi + weighted_interaction
        else:
                return phi
        

class CustomReLUKANNetwork(ReLUKANNetwork):
    def __init__(self, input_channels: int, output_channels: int, init_feature_extractor, layer_hidden: List[int], neurons_hidden: List[int], base_activation: Union[str, List[str]], relu_grid_size: Union[int, List[int]], relu_k: Union[int, List[int]], relu_train_boundary: Union[bool, List[bool]], apply_interactions_layers=None):
        """
        Dziedziczenie z ReLUKANNetwork, z dodaniem możliwości włączenia interakcji w wybranych warstwach.

        :param apply_interactions_layers: lista bool określająca, czy w danej warstwie ma być włączona interakcja.
        """
        # Jeśli apply_interactions_layers nie jest podane, domyślnie ustawiamy na False dla wszystkich warstw
        if apply_interactions_layers is None:
            apply_interactions_layers = [False] * (len(neurons_hidden) + 1)
        assert len(apply_interactions_layers) == len(neurons_hidden) + 1, "apply_interactions_layers must match the number of layers."

        self.apply_interactions_layers = apply_interactions_layers

        # Wywołanie konstruktora klasy bazowej
        super().__init__(input_channels, output_channels, init_feature_extractor, layer_hidden, neurons_hidden, base_activation, relu_grid_size, relu_k, relu_train_boundary)

    def make_relu_layers(self, input_channels, output_channels, neurons_hidden, layer_hiddens, base_activations, relu_grid_sizes, relu_ks, relu_train_boundaries):
        """
        Rozszerzenie make_relu_layers z klasy bazowej o obsługę interakcji.
        """
        layers = []
        for i, (in_features, out_features) in enumerate(zip([input_channels] + neurons_hidden, neurons_hidden + [output_channels])):
            # Sprawdzamy, czy layer_hidden jest typu CustomReLUKANLayer i czy interakcja ma być włączona
            if issubclass(layer_hiddens[i], CustomReLUKANLayer):
                layers.append(
                    layer_hiddens[i](
                        in_features,
                        out_features,
                        base_activations[i],
                        relu_grid_sizes[i],
                        relu_ks[i],
                        relu_train_boundaries[i],
                        apply_interaction=self.apply_interactions_layers[i],  # Dodajemy flagę interakcji
                    )
                )
            else:
                layers.append(
                    layer_hiddens[i](
                        in_features,
                        out_features,
                        base_activations[i],
                        relu_grid_sizes[i],
                        relu_ks[i],
                        relu_train_boundaries[i],
                    )
                )
        return layers



        # interaction_term = torch.einsum('bik,bjk->bij', self.alpha*x, x) #??? torch.dot.outer
        # return phi

        # #Added:
        # # Interactions between variables 
        # print(f"phase high shape: {self.phase_height.shape}") 
        # print(f"alpha shape: {self.alpha.shape}") 
        
        # interaction_term = torch.einsum('bik,bjk->bij', self.alpha*x, x) #??? torch.dot.outer
        # # print(f"interaction_term shape: {interaction_term.shape}") 
        # # interaction_term = torch.einsum('bik,bjk->bij', x, x) #???
        # # interaction_with_alpha = self.alpha * interaction_term  

        # # Summing interaction term across the last dimension and adding to the original computation
        # # phi_star = phi + interaction_with_alpha.sum(dim=-1)
        # phi_star = phi + interaction_term

        # # return phi
        # phi_star.reshape((len(phi_star), self.out_features))



# class CustomReLUKANLayer(ReLUKANLayer):
#     def __init__(self, in_features: int, out_features: int, base_activation: str, grid_size: int, k: int, train_boundaries: bool = True):
#         super(CustomReLUKANLayer, self).__init__(in_features, out_features, base_activation, grid_size, k, train_boundaries)
#         self.alpha = nn.Parameter(torch.ones(out_features, 1))  

#     def forward(self, x: torch.Tensor):
#         x_base = super().forward(x)
#         # print(f"base shape: {x_base.shape}")  # Expected: (batch_size, out_features)

#         interaction = x_base.T @ x_base
#         # print(f"interaction shape: {interaction.shape}")  # Expected: (batch_size, out_features)
#         # print(f"alpha shape: {self.alpha.shape}")  # Expected: (batch_size, out_features)
#         interaction_scaled = self.alpha * interaction
#         interaction_summed = interaction_scaled.sum(dim=-1)
#         return x_base + interaction_summed
    
# class CustomReLUKANLayer(ReLUKANLayer):
#     def __init__(self, in_features: int, out_features: int, base_activation: str, grid_size: int, k: int, train_boundaries: bool = True):
#         super(CustomReLUKANLayer, self).__init__(in_features, out_features, base_activation, grid_size, k, train_boundaries)



#     def forward(self, x: torch.Tensor):
#             x_base = super().forward(x)
#             interaction = x_base.T @ x_base
#             interaction_summed = interaction.sum(dim=-1)
#             return x_base + interaction_summed


# class CustomReLUKANLayer(ReLUKANLayer):
#     def __init__(self, in_features: int, out_features: int, base_activation: str, grid_size: int, k: int, train_boundaries: bool = True):
#         super(CustomReLUKANLayer, self).__init__(in_features, out_features, base_activation, grid_size, k, train_boundaries)



#     def forward(self, x: torch.Tensor):
#             x_base = super().forward(x)
#             interaction = x_base @ x_base.T
#             interaction_summed = interaction.sum(dim=-1)
#             return x_base + interaction_summed.unsqueeze(1) 
    



# #TODO apply decorators for interactions, do not repeat classes

# # Decorator to apply custom interaction
# def apply_custom_interaction(forward_func):
#     @functools.wraps(forward_func)
#     def wrapper(self, x: torch.Tensor):
#         if self.apply_interaction:  
#             # First forward pass through the base layer
#             x_base = forward_func(self, x)
            
#             interaction = x_base @ x_base.T
#             interaction_summed = interaction.sum(dim=-1)
#             return x_base + interaction_summed.unsqueeze(1)
#         else:
#             # If interaction is not applied, return the base forward result
#             return forward_func(self, x)
    
#     return wrapper


# class CustomKANLayer(KANLayer):
#     def __init__(self,      
#         in_features,
#         out_features,
#         base_activation="silu",
#         grid_size=5,
#         spline_order=3,
#         scale_noise=0.1,
#         scale_base=1.0,
#         scale_spline=1.0,
#         enable_standalone_scale_spline=True,
#         grid_eps=0.02,
#         grid_range=[-1, 1]):
#         super(CustomKANLayer, self).__init__(
#             in_features,
#             out_features,
#             base_activation,
#             grid_size,
#             spline_order,
#             scale_noise,
#             scale_base,
#             scale_spline,
#             enable_standalone_scale_spline,
#             grid_eps,
#             grid_range)



#     def forward(self, x: torch.Tensor, update_grid=False):
#         # Call the parent class forward method
#             x_base = super(CustomKANLayer, self).forward(x, update_grid)
#             # print(f"x_base shape: {x_base.shape}")               # Expected: (batch_size, out_features)

#             # Compute interaction term: x @ x.T
#             interaction = x_base @ x_base.T

#             # print(f"interaction shape: {interaction.shape}")  # Expected: (batch_size, out_features)


#             # Scale the interaction term by learnable alpha parameters (element-wise multiplication)
#             # interaction_scaled = self.alpha * interaction

#             # Sum along the last dimension (as per your requirement)
#             # interaction_summed = interaction_scaled.sum(dim=-1)

#             interaction_summed = interaction.sum(dim=-1)

#             # print(f"interaction_summed shape: {interaction_summed.shape}")  # Expected: (batch_size, out_features)

#             #why unsqueeze?
#             #because x_base shape: torch.Size([64, 10])
#             # interaction shape: torch.Size([64, 64])
#             # interaction_summed shape: torch.Size([64])

#             return x_base + interaction_summed.unsqueeze(1) 
           