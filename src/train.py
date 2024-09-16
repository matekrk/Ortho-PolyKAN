import numpy as np
import torch

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
    train_losses, test_losses, train_accs, test_accs = [], [], [], []
    running_losses = []
    model.to(device)
    for e in range(num_epochs):
        model.train()
        for i, batch in enumerate(train_loader):
            X, y = batch
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            running_losses.append(loss.item())
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

    return train_losses, test_losses, train_accs, test_accs, running_losses
