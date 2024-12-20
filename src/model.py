
from baseline import LeNet, ResNet9, Fully_connected_net, Fully_connected_net_bn, Convnet, Convnet_bn
from kan import KANLayer, KANNetwork
from relu import ReLUKANLayer, ReLUKANNetwork
from chebyshev import ChebyLayer
from legendre import LegendreLayer
from polynomial import PolynomialNetwork
from conv import KAN_Convolutional_Layer, KAN_ConvolutionNetwork, KANC_MLP
from ineractions import  CustomReLUKANLayer,CustomReLUKANNetwork

def prepare_model(model_str, **kwargs):
    if kwargs["data_str"] in ["mnist", "fashion"]:
        input_channels = 1
        num_classes = 10
    elif kwargs["data_str"] in ["cifar10", "svhn"]:
        input_channels = 3
        num_classes = 10
    elif kwargs["data_str"] in ["titanic"]:
        input_channels = 9
        num_classes = 2
    elif kwargs["data_str"] in ["toxicity"]:
        input_channels = 1203
        num_classes = 2
    elif kwargs["data_str"] in ["speech_command"]:
        num_classes = 35
        input_channels = 1000
    elif kwargs["data_str"] in ["bean"]:
        num_classes = 7
        input_channels = 16
    elif kwargs["data_str"] in ["income"]:
        num_classes = 2
        input_channels = 14
    if model_str == "baseline_lenet":
        model = LeNet(input_channels, num_classes)
    elif model_str == "resnet9":
        model = ResNet9(input_channels, num_classes, kwargs["softmax"])
    elif model_str == "baseline_fc":
        if kwargs["bn"]:
            model = Fully_connected_net_bn("mnist", num_classes, kwargs["widths"], kwargs["activation"], kwargs["bias"])
        else:
            model = Fully_connected_net("mnist", num_classes, kwargs["widths"], kwargs["activation"], kwargs["bias"])
    elif model_str == "baseline_conv":
        if kwargs["bn"]:
            model = Convnet_bn("mnist", num_classes, kwargs["widths"], kwargs["activation"], kwargs["pooling"], kwargs["bias"])
        else:
            model = Convnet("mnist", num_classes, kwargs["widths"], kwargs["activation"], kwargs["pooling"], kwargs["bias"])
    
    elif model_str.startswith("kan"):

        init_feature_extractor = kwargs["init_feature_extractor"]
        neurons_hidden = kwargs["widths"]
        base_activation = kwargs["activation"]
        layer_norm = kwargs["layer_norm"]
        # apply_interaction =  kwargs["interactions"]
    
        if model_str == "kan_vanilla":
            layer = KANLayer
            model = KANNetwork(input_channels, num_classes, init_feature_extractor, 
                        layer_hidden = layer,
                        neurons_hidden = neurons_hidden,
                        base_activation = base_activation,
                        grid_size = kwargs["grid_size"], 
                        spline_order = kwargs["spline_order"], 
                        scale_noise= kwargs["scale_noise"], 
                        scale_base = kwargs["scale_base"], 
                        scale_spline = kwargs["scale_spline"],
                  
                        )
        elif model_str == "kan_relu":
            layer = ReLUKANLayer
            model = ReLUKANNetwork(input_channels, num_classes, init_feature_extractor, 
                        layer_hidden = layer, 
                        neurons_hidden = neurons_hidden, 
                        base_activation = base_activation,
                        relu_grid_size = kwargs["relu_grid_size"],
                        relu_k = kwargs["relu_k"],
                        relu_train_boundary=kwargs["relu_train_boundary"])
        elif model_str == "kan_relu_interactions":
          
            layer = CustomReLUKANLayer
            model = CustomReLUKANNetwork(input_channels, num_classes, init_feature_extractor, 
                        layer_hidden = layer, 
                        neurons_hidden = neurons_hidden, 
                        base_activation = base_activation,
                        relu_grid_size = kwargs["relu_grid_size"],
                        relu_k = kwargs["relu_k"],
                        relu_train_boundary=kwargs["relu_train_boundary"],
                        apply_interactions_layers= kwargs["relu_apply_interactions"]
                     
                        )

        
        #TODO
        elif model_str == "kan_convolution": 
            # layer = KAN_Convolutional_Layer
            model = KANC_MLP()

        elif model_str in ["kan_polynomial", "kan_chebyshev", "kan_legendre"]:
            if model_str == "kan_polynomial":
                layer == "kan_polynomial"
            elif model_str == "kan_chebyshev":
                layer = ChebyLayer
            elif model_str == "kan_legendre":
                layer = LegendreLayer
            
            model = PolynomialNetwork(input_channels, num_classes, init_feature_extractor, 
                                      layer_hidden = layer,
                                      neurons_hidden = neurons_hidden,
                                      base_activation = base_activation,
                                      polynomial_order = kwargs["polynomial_order"],
                                      layer_norm = layer_norm)
            
    return model
