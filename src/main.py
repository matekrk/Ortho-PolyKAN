import os
from datetime import datetime
import argparse
import torch

from data import prepare_data
from model import prepare_model
from train import prepare_train, train
from evaluate import evaluate
from utils import plot_training, classification_accuracy

def main():
    parser = argparse.ArgumentParser(description="Main method for Ortho PolyKAN")
    
    # Add arguments
    parser.add_argument("model_str", type=str, choices=["baseline_lenet", "baseline_resnet", "baseline_fc", "baseline_conv", "kan_vanilla", "kan_relu", "kan_polynomial", "kan_chebyshev", "kan_legendre"], help="Which architecture")
    parser.add_argument("data_str", type=str, choices=["arithmetic", "conti-arithmetic", "mnist", "fashion", "cifar10", "svhn"], help="Data to train")
    parser.add_argument("--arithmetic_len", type=int, default=1, help="Size of arithmetic dataset")
    parser.add_argument("--arithmetic_id", type=int, default=1, help="For arithmetic dataset which function to choose, in figures visualizations provided")
    parser.add_argument("--arithmetic_dim", type=int, default=1, help="For arithmetic dataset the dimensionality")
    parser.add_argument("--arithmetic_test_shuffle", action="store_true", help="Way to determine arithmetic test set")
    parser.add_argument("--data_path", type=str, default="./data", help="Where to store sets")

    parser.add_argument("--verbose", action="store_true", help="Increase logger verbosity")
    parser.add_argument("--save_model", action="store_true", help="Save model towards the end of training")
    parser.add_argument("--only_eval", action="store_true", help="If true, skip the training")
    parser.add_argument("--cuda", action="store_true", help="If true, then GPU training")
    parser.add_argument("--seed", type=int, default=1997, help="random seed to reproduce")

    parser.add_argument("--objective_str", type=str, choices=["ce", "mse"], default="mse", help="objective to minimize")
    parser.add_argument("--optimizer_str", type=str, choices=["sgd", "adam"], default="adam", help="optimizer to train")
    parser.add_argument("--lr", type=float, default=0.001, help="lr for the optimizer")
    parser.add_argument("--n_epochs", type=int, default=100, help="number of full epochs")
    parser.add_argument("--train_batch_size", type=int, default=64, help="batch size for training")
    parser.add_argument("--test_batch_size", type=int, default=1000, help="batch size for evaluation")

    # baseline
    parser.add_argument("--bn", action="store_true", help="Batch norms for Conv / MLP nets")
    parser.add_argument("--widths", type=int, nargs='+', help="Widths of hidden layers for Conv / MLP")
    parser.add_argument("--activation", type=str, choices=["relu", "hardtanh", "leaky_relu", "selu", "elu", "tanh", "softplus", "sigmoid", "swish", "silu", "gelu"], help="Activation for Conv / MLP nets")
    parser.add_argument("--bias", action="store_true", help="Bias for Conv / MLP nets")
    parser.add_argument("--pooling", type=str, choices=["max", "average"], help="Pooling for Conv")
    parser.add_argument("--softmax", action="store_true", help="Whether last layer is softmax")

    # kan
    parser.add_argument("--init_feature_extractor", action="store_true", help="Start KAN net with CNN block")
    parser.add_argument("--layer_norm", action="store_true", help="whether to use layer norm or batch norm")

    # kan_vaniila
    parser.add_argument("--grid_size", type=int, default=5, help="Vanilla KAN: ")
    parser.add_argument("--spline_order", type=int, default=3, help="Vanilla KAN: ")
    parser.add_argument("--scale_noise", type=float, default=0.1, help="Vanilla KAN: ")
    parser.add_argument("--scale_base", type=float, default=1.0, help="Vanilla KAN: ")
    parser.add_argument("--scale_spline", type=float, default=1.0, help="Vanilla KAN: ")
    # kan_relu
    parser.add_argument("--relu_grid_size", type=int, help="ReLU KAN: grid size")
    parser.add_argument("--relu_k", type=int, help="ReLU KAN: spline degree")
    parser.add_argument("--relu_train_boundary", action="store_true", help="ReLU KAN: train [a,b] boundary parameters")
    # kan_polynomial
    parser.add_argument("--polynomial_order", type=int, help="Polynomial KAN: degree of the polynomials")
    # kan_chebykan
    # kan_legendre
    
    # Parse the arguments
    args = parser.parse_args()
    if args.verbose:
        print(args)
        print(vars(args))
    
    # Use the arguments
    if args.verbose:
        print(f"Data {args.data_str} and model {args.model_str}")
        if not args.only_eval:
            print(f"Experiment: training {args.model_str} on {args.data_str} with {args.optimizer_str} w lr {args.lr}")
        print(f"Experiment: testing")

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    train_loader, test_loader = prepare_data(args.data_str, args.data_path, args.train_batch_size, args.test_batch_size, args.arithmetic_id, args.arithmetic_dim)
    
    model_kwargs = args
    model = prepare_model(**vars(model_kwargs))
    if args.verbose:
        print(f"Model with {sum(p.numel() for p in model.parameters())} parameters")

    compute_accuracy_fn = classification_accuracy
    
    if not args.only_eval and args.verbose:
        print("Processing training")

    optimizer, criterion = prepare_train(model, args.optimizer_str, args.objective_str, args.lr)
    train_losses, test_losses, train_accs, test_accs, running_losses = train(model, train_loader, test_loader, compute_accuracy_fn, criterion, optimizer, args.n_epochs, device, args.verbose)

    if args.verbose:
        print("Processing evaluation")
    evaluate(model, test_loader, criterion, device, compute_accuracy_fn)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("models", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    if args.save_model:
        torch.save(model.state_dict(), f"models/last_model_{timestamp}.pt")

    fig = plot_training(train_losses, test_losses, train_accs, test_accs, running_losses)
    fig.savefig(f"figures/last_training_{timestamp}.png")


if __name__ == "__main__":
    main()
