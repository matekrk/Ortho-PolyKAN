import numpy as np
import torch
import wandb
from evaluate import evaluate

def prepare_train(model, optimizer_str, criterion_str, lr):

    optimizers = {
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD
    }
    optimizer = optimizers[optimizer_str](model.parameters(), lr)
    criterions = {
        "mse": torch.nn.MSELoss(),
        "ce": torch.nn.CrossEntropyLoss()
    }
    criterion = criterions[criterion_str]
    return optimizer, criterion

def train(model, train_loader, test_loader, compute_accuracy_fn, criterion, optimizer, num_epochs, device, verbose):
    train_losses, test_losses, train_accs, test_accs, gradient_norms = [], [], [], [], []
    running_losses = []
    model.to(device)
    for e in range(num_epochs):
        model.train()
        epoch_gradient_norms = []
        for i, batch in enumerate(train_loader):
            X, y = batch
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()


           # Obliczanie normy gradientów i sprawdzenie czy nie eksplodują
            total_norm = 0
            for param in model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)  # Norma L2 gradientu dla danego parametru
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5  # Norma L2 dla wszystkich gradientów
            epoch_gradient_norms.append(total_norm)
            print(f"Gradient norm: {total_norm:.4f}")


            optimizer.step()
            running_losses.append(loss.item())

             # Logowanie po każdej iteracji
            wandb.log({
                "batch/loss": loss.item(),
                "batch/gradient_norm": total_norm
            })

            
            if verbose and i % 10 == 0:
                print(f"Epoch {e} Iter {i} Running loss {np.mean(np.array(running_losses)):.4f}")

        train_loss, train_acc = evaluate(model, train_loader, criterion, device, compute_accuracy_fn)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device, compute_accuracy_fn)
        if verbose:
            print()
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        gradient_norms.append(np.mean(epoch_gradient_norms))

        wandb.log({
            "epoch/train_loss": train_loss,
            "epoch/train_accuracy": train_acc,
            "epoch/test_loss": test_loss,
            "epoch/test_accuracy": test_acc,
            "epoch/gradient_norm": np.mean(epoch_gradient_norms),
            "epoch": e
        })


    return train_losses, test_losses, train_accs, test_accs, running_losses
