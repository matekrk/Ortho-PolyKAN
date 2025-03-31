import os
import time
import argparse
import yaml
import wandb
from datetime import datetime
import torch
from itertools import product
from data import prepare_data
from model import prepare_model
from train import prepare_train, train
from evaluate import evaluate
from utils import plot_training, classification_accuracy


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def run_experiment(config1, wandb_config):
    """Run a single experiment."""
    wandb.init(project=wandb_config["project"], config=config1, 
            tags=[ f"dataset:{config1['data_str']}",
            f"lr:{config1['lr']}",
            f"interactions:{config1.get('relu_apply_interactions', False)}",
            f"widths:{config1.get('widths', [])}",
            f"epochs:{config1.get('n_epochs', [])}",
            f"batch:{config1.get('train_batch_size', [])}",
            f"relu_grid_size:{config1.get('relu_grid_size', [])}",
            f"relu_k:{config1.get('relu_k', [])}"],
            name=f"{config1['data_str']}_lr_{config1['lr']}_int_{config1['relu_apply_interactions']}_wds_{config1['widths']}_grid_{config1['relu_grid_size']}_k_{config1['relu_k']}"
        )
    config = wandb.config

    device = "cuda" if config.cuda and torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_loader, test_loader = prepare_data(config.data_str, "./data", config.train_batch_size, 1000, None, None)

    model = prepare_model(
        init_feature_extractor=config1.get("init_feature_extractor", False), 
        layer_norm=config1.get("layer_norm", False),  
        relu_train_boundary=config1.get("relu_train_boundary", False),  
        **config1
    ).to(device)

    optimizer, criterion = prepare_train(model, config.optimizer_str, config.objective_str, config.lr)

    start_time = time.time()
    
    train_losses, test_losses, train_accs, test_accs, running_losses = train(
        model, train_loader, test_loader, classification_accuracy, criterion, optimizer, config.n_epochs, device, verbose=True
    )
    execution_time = time.time() - start_time

    evaluate(model, test_loader, criterion, device, classification_accuracy)

    wandb.log({
        "execution_time": execution_time
    })

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), f"models/last_model_{timestamp}.pt")

    wandb.finish()


def hypertune():
    parser = argparse.ArgumentParser(description="Main method for Ortho PolyKAN")
    parser.add_argument("--grid_config", type=str, help="Path to hyperparameter grid config file (.yaml)", required=True)
    parser.add_argument("--wandb_config", type=str, help="Path to wandb config file (.yaml)", required=True)

    args = parser.parse_args()

    wandb_config = load_config(args.wandb_config)
    wandb.login(key=wandb_config["api_key"])

    param_grid = load_config(args.grid_config)

    # Pobieramy wartości hiperparametrów (jeśli klucza nie ma, używamy domyślnej wartości)
    data_str_values = param_grid.get("data_str", ["mnist"])
    lr_values = param_grid.get("lr", [0.001])
    n_epochs_values = param_grid.get("n_epochs", [10])
    train_batch_size_values = param_grid.get("train_batch_size", [64])
    relu_grid_size_values = param_grid.get("relu_grid_size", [3])
    relu_k_values = param_grid.get("relu_k", [2])
    relu_apply_interactions_values = param_grid.get("relu_apply_interactions", [[True, False, True]])
    activation_values = param_grid.get("base_activation", ["relu"])
    widths_values = param_grid.get("widths", [[100, 10]])

    # Tworzymy wszystkie możliwe kombinacje hiperparametrów
    for params in product(
        data_str_values, lr_values, n_epochs_values, train_batch_size_values,
        relu_grid_size_values, relu_k_values, relu_apply_interactions_values, activation_values, widths_values
    ):
        data_str, lr, n_epochs, train_batch_size, relu_grid_size, relu_k, relu_apply_interactions, activation, widths = params

        config = {
            "model_str": param_grid.get("model_str", "kan_relu"),
            "data_str": data_str,
            "lr": lr,
            "n_epochs": n_epochs,
            "train_batch_size": train_batch_size,
            "cuda": param_grid.get("cuda", True),
            "relu_grid_size": relu_grid_size,
            "relu_k": relu_k,
            "relu_apply_interactions": relu_apply_interactions,
            "widths": widths,  
            "optimizer_str": param_grid.get("optimizer_str", "adam"),
            "objective_str": param_grid.get("objective_str", "ce"),
            "activation": activation
        }

        print(f"\n Running experiment with: {config}\n")
        run_experiment(config, wandb_config)


if __name__ == "__main__":
    hypertune()
