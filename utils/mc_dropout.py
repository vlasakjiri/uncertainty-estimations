# utilities for monte Carlo Dropout

from typing import OrderedDict
import torch

# taken from https://github.com/mattiasegu/uncertainty_estimation_deep_learning


def set_training_mode_for_dropout(model, training=True):
    """Set Dropout mode to train or eval."""

    for m in model.modules():
        #        print(m.__class__.__name__)
        if m.__class__.__name__.startswith('Dropout'):
            if training == True:
                m.train()
            else:
                m.eval()
    return model


def set_dropout_p(model, block, prob, omitted_blocks=[]):
    """Set dropout probability for dropout and dropout2d layers"""
    for name, p in block.named_children():
        if any(map(lambda x: isinstance(p, x), omitted_blocks)):
            continue
        if isinstance(p, torch.nn.Module):
            set_dropout_p(model, p, prob, omitted_blocks)
        if isinstance(p, torch.nn.Dropout):
            setattr(block, name, torch.nn.Dropout(p=prob))
            return model
        elif isinstance(p, torch.nn.Dropout2d):
            setattr(block, name, torch.nn.Dropout2d(p=prob))
            return model


def mc_dropout(model, X, output_shape, T=40):
    """Perform Monte Carlo Dropout sampling"""
    model.eval()
    set_training_mode_for_dropout(model)
    out = torch.zeros(T, X.shape[0], *output_shape)
    for i in range(T):
        with torch.no_grad():
            x = model(X)
            if isinstance(x, OrderedDict):
                out[i] = x["out"].cpu()
            else:
                out[i] = x.cpu()
    probs = out.softmax(dim=2)
    means = probs.mean(dim=0)
    vars = probs.var(dim=0)
    model.eval()
    return means, vars


def add_dropout(model, block, prob, add_after=torch.nn.ReLU, dropout_cls=torch.nn.Dropout2d, omitted_blocks=[]):
    """Add dropout layers after specified layer"""
    for name, p in block.named_children():
        if any(map(lambda x: isinstance(p, x), omitted_blocks)):
            continue
        if isinstance(p, torch.nn.Module) or isinstance(p, torch.nn.Sequential):
            add_dropout(model, p, prob, add_after, dropout_cls, omitted_blocks)

        if isinstance(p, add_after):
            setattr(block, name, torch.nn.Sequential(
                add_after(), dropout_cls(p=prob)))
