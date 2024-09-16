import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch

def get_activation(activation: str):
    if activation == 'relu':
        return torch.nn.ReLU()
    elif activation == 'hardtanh':
        return torch.nn.Hardtanh()
    elif activation == 'leaky_relu':
        return torch.nn.LeakyReLU()
    elif activation == 'selu':
        return torch.nn.SELU()
    elif activation == 'elu':
        return torch.nn.ELU()
    elif activation == "tanh":
        return torch.nn.Tanh()
    elif activation == "softplus":
        return torch.nn.Softplus()
    elif activation == "sigmoid":
        return torch.nn.Sigmoid()
    elif activation == "swish":
        return torch.nn.SiLU()
    elif activation == "gelu":
        return torch.nn.GELU()
    else:
        raise NotImplementedError("unknown activation function: {}".format(activation))

def get_pooling(pooling: str):
    if pooling == 'max':
        return torch.nn.MaxPool2d((2, 2))
    elif pooling == 'average':
        return torch.nn.AvgPool2d((2, 2))

def classification_accuracy(pred, y):
    _, predicted = torch.max(pred.data, 1)
    _, label = torch.max(y.data, 1)
    return (predicted == label).sum().item() / len(pred)


def plot_training(train_losses, test_losses, train_accuracies, test_accuracies, running_losses = None):
    
    nrows = 2 if running_losses is not None else 1
    fig, axs = plt.subplots(nrows, 2, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 2], 'width_ratios': [1, 1]} if nrows == 2 else None)

    axs[0, 0].plot(train_losses, marker='o', linestyle='-', color='r', label="train")
    axs[0, 0].plot(test_losses, marker='o', linestyle='-', color='g', label="test")
    axs[0, 0].set_title('Loss')
    axs[0, 0].set_xlabel('Iteration')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()

    axs[0, 1].plot(train_accuracies, marker='o', linestyle='-', color='r', label="train")
    axs[0, 1].plot(test_accuracies, marker='o', linestyle='-', color='g', label="test")
    axs[0, 1].set_title('Accuracy')
    axs[0, 1].set_xlabel('Iteration')
    axs[0, 1].set_ylabel('Accuracy')
    axs[0, 1].legend()

    if nrows == 2:
        axs[1, 0].plot(running_losses, color='r', label="train")
        axs[1, 0].set_title('Loss')
        axs[1, 0].set_xlabel('Iteration')
        axs[1, 0].set_ylabel('Loss')
        axs[1, 0].legend()

        axs[1, 1].remove() #FIXME: or ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)

    plt.tight_layout()
    return fig


def plot_aggregate(experiments):

    # Example data
    experiments = {
        0.01: {'train_losses': [0.6, 0.5], 'test_losses': [0.6, 0.65], 'train_accuracies': [0.8, 0.9], 'test_accuracies': [0.8, 0.9]},
    }

    markers = ["x", "o", "^", "s", "*"]
    colors = ["r", "g", "b", "y", "c"]

    # Extract data for plotting
    # train_losses = [exp['train_loss'] for exp in experiments]
    # test_losses = [exp['test_loss'] for exp in experiments]
    # accuracies = [exp['accuracy'] for exp in experiments]

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    for i, name in enumerate(experiments.keys()):
        axs[0, 0].plot(experiments[name]["train_losses"], marker='o', linestyle='-', color=colors[i])
    axs[0, 0].set_title('Train Loss')
    axs[0, 0].set_xlabel('Iteration')
    axs[0, 0].set_ylabel('Loss')

    for i, name in enumerate(experiments.keys()):
        axs[0, 1].plot(experiments[name]["test_losses"], marker='o', linestyle='-', color=colors[i])
    axs[0, 1].set_title('Test Loss')
    axs[0, 1].set_xlabel('Iteration')
    axs[0, 1].set_ylabel('Loss')

    for i, name in enumerate(experiments.keys()):
        axs[1, 0].plot(experiments[name]["train_accuracies"], marker='o', linestyle='-', color=colors[i])
    axs[1, 0].set_title('Accuracy')
    axs[1, 0].set_xlabel('Iteration')
    axs[1, 0].set_ylabel('Accuracy')

    for i, name in enumerate(experiments.keys()):
        axs[1, 0].plot(experiments[name]["test_accuracies"], marker='o', linestyle='-', color=colors[i])
    axs[1, 0].set_title('Accuracy')
    axs[1, 0].set_xlabel('Iteration')
    axs[1, 0].set_ylabel('Accuracy')

    plt.tight_layout()
    return fig

def test_mul(model_kan):
    # model_kan = KAN([2, 2, 1], base_activation=torch.nn.Identity)
    optimizer = torch.optim.LBFGS(model_kan.parameters(), lr=1)
    with tqdm(range(100)) as pbar:
        for i in pbar:
            loss, reg_loss = None, None

            def closure():
                optimizer.zero_grad()
                x = torch.rand(1024, 2)
                y = model_kan(x, update_grid=(i % 20 == 0))

                assert y.shape == (1024, 1)
                nonlocal loss, reg_loss
                u = x[:, 0]
                v = x[:, 1]
                loss = torch.nn.functional.mse_loss(y.squeeze(-1), (u + v) / (1 + u * v))
                reg_loss = model_kan.regularization_loss(1, 0)
                (loss + 1e-5 * reg_loss).backward()
                return loss + reg_loss

            optimizer.step(closure)
            pbar.set_postfix(mse_loss=loss.item(), reg_loss=reg_loss.item())
    for layer in model_kan.layers:
        print(layer.spline_weight)

def show_base(model_rk, phase_num, step):
    # rk = ReLUKANLayer(1, phase_num, step, 1)
    x = torch.Tensor([np.arange(-600, 1024+600) / 1024]).T
    x1 = torch.relu(x - model_rk.phase_low)
    x2 = torch.relu(model_rk.phase_height - x)
    y = x1 * x1 * x2 * x2 * model_rk.r * model_rk.r
    for i in range(phase_num+step):
        plt.plot(x, y[:, i:i+1].detach(), color='black')
    plt.show()
    print('1')
