import os
import random
import numpy as np
import torch

def set_seed(seed: int, 
             deterministic: bool = True):
    """
    Sets random, numpy, and PyTorch seeds.
    
    Parameters
    ----------
    
    seed
        The seed value.
    
    Returns
    -------
    
    nuthin !
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)