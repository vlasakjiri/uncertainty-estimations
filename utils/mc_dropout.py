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
    softmax = torch.nn.Softmax(dim=1)
    out = torch.zeros(T, X.shape[0], num_classes)
    for i in range(T):
        with torch.no_grad():
            out[i] = softmax(model(X).cpu())
    return out
