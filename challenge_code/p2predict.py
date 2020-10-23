from argparse import ArgumentParser

import numpy as np
import torch
from pytorch_lightning import Trainer

from .p2model import Challenger

np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main(args):
    model = Challenger.load_from_checkpoint(args.checkpoint_path, hparam_overrides=args)
    model.hparams.logger = None
    model.hparams.checkpoint_callback = None
    model.prepare_data()
    trainer = Trainer.from_argparse_args(args)
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
    
    
    args = parser.parse_args()
    main(args)
