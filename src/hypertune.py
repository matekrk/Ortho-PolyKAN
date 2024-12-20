import os
import time
import argparse
import yaml
import wandb
from datetime import datetime
import torch
from data import prepare_data
from model import prepare_model
from train import prepare_train, train
from evaluate import evaluate
from utils import plot_training, classification_accuracy


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def str_to_bool_list(value):
    """Convert a comma-separated string into a list of booleans."""
    return [v.strip().lower() == 'true' for v in value.split(',')]


def run_experiment(config1, wandb_config):
    """Run a single experiment."""
    wandb.init(project=wandb_config["project"], config=config1, 
            tags=[ f"dataset:{config1['data_str']}",
            f"lr:{config1['lr']}",
            f"interactions:{config1.get('relu_apply_interactions', False)}",
            f"widths:{config1.get('widths', [])}"],
            name=f"{config1['data_str']}_lr_{config1['lr']}_int_{config1['relu_apply_interactions']}_wds_{config1['widths']}"
        )
    config = wandb.config
    print(config)
  
    # device = "cuda" if config["cuda"] and torch.cuda.is_available() else "cpu"

    device = "cuda" if config.cuda and torch.cuda.is_available() else "cpu"

    # train_loader, test_loader = prepare_data(config["data_str"], "./data", config["train_batch_size"], 1000, None, None)

    train_loader, test_loader = prepare_data(config.data_str, "./data", config.train_batch_size, 1000, None, None)
    
 
    model = prepare_model(init_feature_extractor=config1.get("init_feature_extractor", False), 
                          layer_norm=config1.get("layer_norm", False),  
                          relu_train_boundary=config1.get("relu_train_boundary", False),  
                          **config1).to(device)

    # model = prepare_model(init_feature_extractor=config.init_feature_extractor, 
    #                       layer_norm=config.layer_norm,  
    #                       relu_train_boundary=config.relu_train_boundary, 
    #                       **vars(config)).to(device)
    # model = prepare_model(**vars(config)).to(device)

    # optimizer, criterion = prepare_train(model, config["optimizer_str"], config["objective_str"], config["lr"])

    optimizer, criterion = prepare_train(model, config.optimizer_str, config.objective_str, config.lr)

    start_time = time.time()
    
    train_losses, test_losses, train_accs, test_accs, running_losses = train(
        # model, train_loader, test_loader, classification_accuracy, criterion, optimizer, config["n_epochs"], device, verbose=True

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
    # Optional config files
    # parser.add_argument("--config", type=str, help="Path to experiment config file (.yaml)")
    parser.add_argument("--grid_config", type=str, help="Path to hyperparameter grid config file (.yaml)")
    parser.add_argument("--wandb_config", type=str, help="Path to hyperparameter wandb config file (.yaml)")

    args = parser.parse_args()

    wandb_config = load_config(args.wandb_config)
    wandb.login(key=wandb_config["api_key"])


    if not args.grid_config:
        raise ValueError("Please provide --grid_config for tuning mode")


    param_grid = load_config(args.grid_config)

    for data_str in param_grid.get("data_str", ["mnist"]):
                for lr in param_grid.get("lr", [0.001]):
                    for n_epochs in param_grid.get("n_epochs", [10]):
                        for train_batch_size in param_grid.get("train_batch_size", [64]):
                            for relu_grid_size in param_grid.get("relu_grid_size", [3]):
                                for relu_k in param_grid.get("relu_k", [2]):
                                    for relu_apply_interactions in param_grid.get("relu_apply_interactions", [[True, False, True]]):
                                        for activation in param_grid.get("base_activation"):
                                            for widths in param_grid.get("widths", [[100,10]]):  
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
                                            run_experiment(config, wandb_config)
    # else:
 
    #     if args.config:
    #         config = load_config(args.config1)
    #     else:
    #         raise ValueError("Please provide --config for single experiment mode")

    #     run_experiment(config)


if __name__ == "__main__":
    hypertune()
