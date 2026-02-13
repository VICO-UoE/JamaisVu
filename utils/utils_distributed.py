import torch
from torch.nn.parallel import DistributedDataParallel as DDP

def get_model_attribute(model, attr_name):
    """Helper function to get an attribute from the model, handling DDP wrapping."""
    if isinstance(model, DDP):
        return getattr(model.module, attr_name)
    else:
        return getattr(model, attr_name)