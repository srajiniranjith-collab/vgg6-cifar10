"""Optimizer factory for the six optimizers studied in Q2(b)."""
import torch.optim as optim

OPTIMIZERS = ["sgd", "nesterov", "adam", "adagrad", "rmsprop", "nadam"]


def build_optimizer(name: str, params, lr: float, weight_decay: float = 0.0,
                    momentum: float = 0.9):
    """`momentum` is used by sgd / nesterov / rmsprop. Adam-family and Adagrad ignore it."""
    name = name.lower()
    if name == "sgd":
        return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "nesterov":
        return optim.SGD(params, lr=lr, momentum=momentum, nesterov=True,
                         weight_decay=weight_decay)
    if name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adagrad":
        return optim.Adagrad(params, lr=lr, weight_decay=weight_decay)
    if name == "rmsprop":
        return optim.RMSprop(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "nadam":
        return optim.NAdam(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer {name!r}; choose from {OPTIMIZERS}")
