from argparse import ArgumentParser

import numpy as np
import torch
from pytorch_lightning import Trainer, seed_everything
import glob

from .model import Challenger

def main(hparams):
    if hparams.seed is not None:
        seed = hparams.seed
    else:
        np.random.randint(1, high=10000, size=1)[0]
    seed_everything(seed)
    
    model = Challenger.load_from_checkpoint(hparams.checkpoint_path, hparams=hparams)
    model.hparams.logger = None
    model.hparams.checkpoint_callback = None
    model.prepare_data()
    trainer = Trainer.from_argparse_args(hparams)
    trainer.test(model)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser = Challenger.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    parser.add_argument("root_directory", type=str,
                        help="Directory containing images and segmentation masks.")
    
    parser.add_argument("clinical_data_path", type=str,
                        help="Path to CSV file containing the clinical data.")
    
    parser.add_argument("checkpoint_path", type=str,
                        help="Path to saved model checkpoint.")
    
    parser.add_argument("--pred_save_path", type=str, default="data/predictions/phase2.npy",
                        help="Directory where final predictions will be saved.")
    
    parser.add_argument("--cache_dir", type=str, default="./data/data_cache",
                        help=("Directory where the preprocessed data will be saved."))
    
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of worker processes to use for data loading.")
    
    parser.add_argument("--exp_name", type=str, default="challenger",
                        help="Experiment name for logging purposes.")
    
    parser.add_argument ("--design", type=str, default="aaai_cnn",
                        help="Choose architecture design")
    
    parser.add_argument ("--seed", type=int, default=None,
                        help="Choose architecture design")
    
    hparams = parser.parse_args()
    main(hparams)
