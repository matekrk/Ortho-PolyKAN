import os
from datetime import datetime
import argparse
import torch

from data import prepare_data
from model import prepare_model
from train import prepare_train, train
from evaluate import evaluate
from utils import plot_training, classification_accuracy
import time

def str_to_bool_list(value):
    """Convert a comma-separated string into a list of booleans."""
    return [v.strip().lower() == 'true' for v in value.split(',')]



def compare():
    
    parser = argparse.ArgumentParser(description="Main method for Ortho PolyKAN")
    
    # Argumenty wspólne dla obu modeli
    parser.add_argument("--data_path", type=str, default="./data", help="Where to store sets")
    parser.add_argument("--cuda", action="store_true", help="If true, then GPU training")
    parser.add_argument("--n_epochs", type=int, default=100, help="Number of full epochs")
    parser.add_argument("--train_batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--test_batch_size", type=int, default=1000, help="Batch size for evaluation")
    parser.add_argument("--optimizer_str", type=str, choices=["sgd", "adam"], default="adam", help="Optimizer to train")
    parser.add_argument("--objective_str", type=str, choices=["ce", "mse"], default="ce", help="Objective to minimize")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--widths", type=int, nargs='+', help="Widths of hidden layers")
    parser.add_argument("--activation", type=str, default="relu", help="Activation for the network")
    parser.add_argument("--relu_grid_size", type=int, help="ReLU KAN: grid size")
    parser.add_argument("--relu_k", type=int, help="ReLU KAN: spline degree")
    parser.add_argument("--relu_train_boundary", action="store_true", help="ReLU KAN: train [a,b] boundary parameters")
    parser.add_argument("--relu_apply_interactions", type=str_to_bool_list, help="Comma-separated list of booleans for interactions")
    parser.add_argument("--init_feature_extractor", action="store_true", help="Start KAN net with CNN block")
    parser.add_argument("--layer_norm", action="store_true", help="whether to use layer norm or batch norm")
    # Dataset i model
    parser.add_argument("--data_str", type=str, default="mnist", help="Dataset to use")
    parser.add_argument("--verbose", action="store_true", help="Increase logger verbosity")

    # Parse arguments
    args = parser.parse_args()

    if args.verbose:
        print(f"Arguments: {vars(args)}")

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    # Przygotowanie danych
    train_loader, test_loader = prepare_data(args.data_str, args.data_path, args.train_batch_size, args.test_batch_size)
    dataset = (train_loader, test_loader)

    model_kwargs = args
    # common_params = {
    #     "data_str": args.data_str,
    #     "init_feature_extractor": False,
    #     "neurons_hidden": args.widths,
    #     "base_activation": args.activation,
    #     "relu_grid_size": args.relu_grid_size,
    #     "relu_k": args.relu_k,
    #     "relu_train_boundary": args.relu_train_boundary,
    #     "widths": args.widths,
    #     "activation": args.activation
    # }


    model1_params = {**vars(model_kwargs), "apply_interactions_layers": args.relu_apply_interactions}
    model2_params = {**vars(model_kwargs)}


    results_model1 = train_model("kan_relu_interactions", model1_params, dataset, args, device)
    results_model2 = train_model("kan_relu", model2_params, dataset, args, device)

    # Tworzenie wykresów porównawczych
    fig = compare_models_and_plot_kan(
        results_model1=results_model1,
        results_model2=results_model2,
        dataset_name=args.data_str,
        execution_time=max(results_model1["execution_time"], results_model2["execution_time"]),
        interaction_layers=args.relu_apply_interactions)

 
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("figures", exist_ok=True)
    fig.savefig(f"figures/{args.data_str}_comparison_{timestamp}.png")


def train_model(model_str, model_params, dataset, args, device):

    model = prepare_model(model_str, **model_params)
    optimizer, criterion = prepare_train(model, args.optimizer_str, args.objective_str, args.lr)

    if args.verbose:
        print(f"Training model {model_str}")

    start_time = time.time()
    train_losses, test_losses, train_accs, test_accs, running_losses = train(
        model, dataset[0], dataset[1],
        classification_accuracy,
        criterion,
        optimizer,
        args.n_epochs,
        device,
        args.verbose
    )
    execution_time = time.time() - start_time

    return {
        "train_losses": train_losses,
        "test_losses": test_losses,
        "train_accuracies": train_accs,
        "test_accuracies": test_accs,
        "running_losses": running_losses,
        "execution_time": execution_time
    }

import matplotlib.pyplot as plt

def compare_models_and_plot_kan(results_model1, results_model2, dataset_name, execution_time, interaction_layers=None):
    """
    Compare two models and plot training and testing results.
    
    Args:
        results_model1 (dict): Results from the first model (with interactions).
        results_model2 (dict): Results from the second model (without interactions).
        dataset_name (str): Name of the dataset.
        execution_time (float): Execution time of the training.
        interaction_layers (list of int, optional): List of layers where interactions are applied.

    Returns:
        matplotlib.figure.Figure: The figure with the comparison plots.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Comparison of ReLU models on {dataset_name} dataset", fontsize=16)

    # Training Losses
    axes[0, 0].plot(results_model1["train_losses"], label="Train Loss (Interactions)", color="blue")
    axes[0, 0].plot(results_model2["train_losses"], label="Train Loss (No Interactions)", color="orange")
    axes[0, 0].set_title("Training Loss")
    axes[0, 0].set_xlabel("Epochs")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid()

    # Test Losses
    axes[0, 1].plot(results_model1["test_losses"], label="Test Loss (Interactions)", color="blue")
    axes[0, 1].plot(results_model2["test_losses"], label="Test Loss (No Interactions)", color="orange")
    axes[0, 1].set_title("Test Loss")
    axes[0, 1].set_xlabel("Epochs")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend()
    axes[0, 1].grid()

    # Training Accuracy
    axes[1, 0].plot(results_model1["train_accuracies"], label="Train Accuracy (Interactions)", color="blue")
    axes[1, 0].plot(results_model2["train_accuracies"], label="Train Accuracy (No Interactions)", color="orange")
    axes[1, 0].set_title("Training Accuracy")
    axes[1, 0].set_xlabel("Epochs")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].legend()
    axes[1, 0].grid()

    # Test Accuracy
    axes[1, 1].plot(results_model1["test_accuracies"], label="Test Accuracy (Interactions)", color="blue")
    axes[1, 1].plot(results_model2["test_accuracies"], label="Test Accuracy (No Interactions)", color="orange")
    axes[1, 1].set_title("Test Accuracy")
    axes[1, 1].set_xlabel("Epochs")
    axes[1, 1].set_ylabel("Accuracy")
    axes[1, 1].legend()
    axes[1, 1].grid()

    # Add execution time
    fig.text(0.5, 0.01, f"Execution Time: {execution_time:.2f} seconds", ha="center", fontsize=12)

    # If interaction layers are provided, add a note
    if interaction_layers is not None:
        interaction_info = f"Interaction in layer(s): {', '.join(map(str, interaction_layers))}"
        fig.text(0.5, 0.94, interaction_info, ha="center", fontsize=12, style='italic')

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


# def compare_models_and_plot_kan(results_model1, results_model2, dataset_name, execution_time):

#     fig, axes = plt.subplots(2, 2, figsize=(16, 12))
#     fig.suptitle(f"Comparison of ReLU models on {dataset_name} dataset", fontsize=16)

#     # Straty treningowe
#     axes[0, 0].plot(results_model1["train_losses"], label="Train Loss (Interactions)", color="blue")
#     axes[0, 0].plot(results_model2["train_losses"], label="Train Loss (No Interactions)", color="orange")
#     axes[0, 0].set_title("Training Loss")
#     axes[0, 0].set_xlabel("Epochs")
#     axes[0, 0].set_ylabel("Loss")
#     axes[0, 0].legend()
#     axes[0, 0].grid()

#     # Straty testowe
#     axes[0, 1].plot(results_model1["test_losses"], label="Test Loss (Interactions)", color="blue")
#     axes[0, 1].plot(results_model2["test_losses"], label="Test Loss (No Interactions)", color="orange")
#     axes[0, 1].set_title("Test Loss")
#     axes[0, 1].set_xlabel("Epochs")
#     axes[0, 1].set_ylabel("Loss")
#     axes[0, 1].legend()
#     axes[0, 1].grid()

#     # Dokładność treningowa
#     axes[1, 0].plot(results_model1["train_accuracies"], label="Train Accuracy (Interactions)", color="blue")
#     axes[1, 0].plot(results_model2["train_accuracies"], label="Train Accuracy (No Interactions)", color="orange")
#     axes[1, 0].set_title("Training Accuracy")
#     axes[1, 0].set_xlabel("Epochs")
#     axes[1, 0].set_ylabel("Accuracy")
#     axes[1, 0].legend()
#     axes[1, 0].grid()

#     # Dokładność testowa
#     axes[1, 1].plot(results_model1["test_accuracies"], label="Test Accuracy (Interactions)", color="blue")
#     axes[1, 1].plot(results_model2["test_accuracies"], label="Test Accuracy (No Interactions)", color="orange")
#     axes[1, 1].set_title("Test Accuracy")
#     axes[1, 1].set_xlabel("Epochs")
#     axes[1, 1].set_ylabel("Accuracy")
#     axes[1, 1].legend()
#     axes[1, 1].grid()

#     # Dodanie czasu wykonania do opisu wykresu
#     fig.text(0.5, 0.01, f"Execution Time: {execution_time:.2f} seconds", ha="center", fontsize=12)

#     fig.tight_layout(rect=[0, 0.03, 1, 0.95])
#     return fig


if __name__ == "__main__":
    compare()