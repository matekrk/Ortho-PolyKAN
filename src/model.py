
from baseline import LeNet, Fully_connected_net, Fully_connected_net_bn, Convnet, Convnet_bn
from cheby import MNISTChebyKAN
from kan import KAN
from legendre import LegendreKAN
from relukan import ReLUKAN

def prepare_model_mnist(model_str, **kwargs):
    num_classes = 10
    if model_str == "baseline_lenet":
        model = LeNet(num_classes = 10)
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
    elif model_str == "kan_cheby":
        model = MNISTChebyKAN()
    elif model_str == "kan_vanilla":
        model = KAN(layers_hidden = kwargs["layers_hidden"], 
                    grid_size = kwargs["grid_size"], 
                    spline_order = kwargs["spline_order"], 
                    scale_noise= kwargs["scale_noise"], 
                    scale_base = kwargs["scale_base"], 
                    scale_spline = kwargs["scale_spline"])
    elif model_str == "kan_legendre":
        pass
    elif model_str == "kan_relu":
        model = ReLUKAN(width = kwargs["width"],
                        grid = kwargs["grid"],
                        k = kwargs["k"])
    return model

def prepare_model(data_str, model_str, **kwargs):
    if data_str == "mnist":
        return prepare_model_mnist(model_str, **kwargs)
