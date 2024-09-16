
def evaluate(model, test_loader, criterion, device, compute_accuracy_fn = None):
    running_loss, running_acc, total_samples = 0.0, 0.0, 0
    model.eval()
    for i, batch in enumerate(test_loader):
        X, y = batch
        X, y = X.to(device), y.to(device)
        pred = model(X)
        running_loss += criterion(pred, y).item()
        if compute_accuracy_fn is not None:
            running_acc += compute_accuracy_fn(pred, y) * len(X)
            total_samples += len(X)

    running_loss = running_loss / len(test_loader) if len(test_loader) else 0.0
    running_acc = running_acc / total_samples if total_samples else 0.0
    return running_loss, running_acc
