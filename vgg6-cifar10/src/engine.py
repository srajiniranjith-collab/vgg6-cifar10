"""Training and evaluation loops (kept separate from data/model code)."""
import torch


def train_one_epoch(model, loader, criterion, optimizer, device):
    """One pass over `loader`. Returns (mean_loss, accuracy %).

    NOTE: model.train() is called here on every epoch. evaluate() switches the
    model to eval mode, so it must be switched back each epoch (the template
    only called it once, which left BatchNorm in eval mode after epoch 0).
    Accuracy is the running accuracy on augmented batches (cheaper than a
    separate pass over the train set).
    """
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        n = y.size(0)
        loss_sum += loss.item() * n
        correct += out.argmax(1).eq(y).sum().item()
        total += n
    return loss_sum / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Returns (mean_loss, top-1 accuracy %) on `loader`."""
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        loss_sum += criterion(out, y).item() * y.size(0)
        correct += out.argmax(1).eq(y).sum().item()
        total += y.size(0)
    return loss_sum / total, 100.0 * correct / total
