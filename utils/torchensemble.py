# Fix for the torchensemble module when loading cuda saved file on cpu

import torch
import os


def load(model, device, save_dir="./", logger=None):
    """Implement model deserialization from the specified directory."""
    if not os.path.exists(save_dir):
        raise FileExistsError("`{}` does not exist".format(save_dir))

    # Decide the base estimator name
    if isinstance(model.base_estimator_, type):
        base_estimator_name = model.base_estimator_.__name__
    else:
        base_estimator_name = model.base_estimator_.__class__.__name__

    # {Ensemble_Model_Name}_{Base_Estimator_Name}_{n_estimators}
    filename = "{}_{}_{}_ckpt.pth".format(
        type(model).__name__,
        base_estimator_name,
        model.n_estimators,
    )
    save_dir = os.path.join(save_dir, filename)

    if logger:
        logger.info("Loading the model from `{}`".format(save_dir))

    state = torch.load(save_dir, map_location=device)
    n_estimators = state["n_estimators"]
    model_params = state["model"]
    model._criterion = state["_criterion"]

    # Pre-allocate and load all base estimators
    for _ in range(n_estimators):
        model.estimators_.append(model._make_estimator())
    model.load_state_dict(model_params)
