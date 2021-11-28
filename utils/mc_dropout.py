import torch

import utils.model


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
            # bn = torch.nn.BatchNorm2d(p.num_features)
            # bn.load_state_dict(p.state_dict())
            # setattr(block, name, bn)


def mc_dropout(model, X, T=40):
    num_classes = utils.model.get_number_of_classes(model)
    model.eval()
    set_training_mode_for_dropout(model)
    out = torch.zeros(T, X.shape[0], num_classes)
    for i in range(T):
        with torch.no_grad():
            out[i] = model(X).cpu()
    probs = out.softmax(dim=2)
    means = probs.mean(dim=0)
    vars = probs.var(dim=0)
    return means, vars


def compute_log_likelihood(y_pred, y_true, sigma):
    dist = torch.distributions.normal.Normal(loc=y_pred, scale=sigma)
    log_likelihood = dist.log_prob(y_true)
    log_likelihood = torch.mean(log_likelihood, dim=1)
    return log_likelihood
