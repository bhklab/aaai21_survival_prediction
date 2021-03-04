from argparse import ArgumentParser

import numpy as np
import torch
from pytorch_lightning import Trainer, seed_everything
import glob

from .p2model import Challenger

def best_ckpt (arr):
    """
    Returns checkpoint with highest ROC_AUC or highest LOSS
    """
    best_idx, best_epoch, best_loss, best_auc = 0, 0, 0, 0    
    
    for idx, c in enumerate(arr):
        epoch, loss, auc = c.split("/")[-1].split("-")
        epoch = float(epoch.split("=")[-1])
        loss  = float(loss.split("=")[-1])
        auc   = float(auc.split("=")[-1][:-5])
        if auc > best_auc:
            best_idx, best_epoch, best_loss, best_auc = idx, epoch, loss, auc
        elif auc == best_auc:
            if loss < best_loss:
                best_idx, best_epoch, best_loss, best_auc = idx, epoch, loss, auc
            elif loss == best_loss:
                if epoch > best_epoch:
                    best_idx, best_epoch, best_loss, best_auc = idx, epoch, loss, auc

    return arr[best_idx]

def main(hparams):
    print("start")
    #print(hparams)
    if hparams.seed is not None:
        seed = hparams.seed
    else:
        np.random.randint(1, high=10000, size=1)[0]
    print(seed)
    seed_everything(seed)
    
    models = sorted(glob.glob(f"/cluster/projects/radiomics/Temp/sejin/aaai21_survival_prediction/data/logs/aaai/{hparams.design}_169*"))

    best_checkpoints = []
    for model in models:
        ckpts = glob.glob(model+"/checkpoints/*")       
        if len(ckpts) > 0:
            best_checkpoints.append(best_ckpt(ckpts))
        
    print(best_checkpoints)
    
    for ckpt in best_checkpoints:
        print("now testing:", ckpt[81:])
        try:
            model = Challenger.load_from_checkpoint(ckpt, hparams=hparams)
            model.hparams.logger = None
            model.hparams.checkpoint_callback = None
            model.prepare_data()
            print(model)
            trainer = Trainer.from_argparse_args(hparams)
            trainer.test(model)
        except RuntimeError as e:
            print(e)
            print(ckpt[81:], ": FAILED\n\n")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser = Challenger.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    parser.add_argument("root_directory", type=str,
                        help="Directory containing images and segmentation masks.")
    
    parser.add_argument("clinical_data_path", type=str,
                        help="Path to CSV file containing the clinical data.")
    
    # parser.add_argument("checkpoint_path", type=str,
    #                     help="Path to saved model checkpoint.")
    
    parser.add_argument("--pred_save_path", type=str, default="data/predictions/phase2.npy",
                        help="Directory where final predictions will be saved.")
    
    parser.add_argument("--cache_dir", type=str, default="./data/data_cache",
                        help=("Directory where the preprocessed data will be saved."))
    
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of worker processes to use for data loading.")
    
    parser.add_argument("--exp_name", type=str, default="challenger",
                        help="Experiment name for logging purposes.")
    
    parser.add_argument ("--design", type=str, default="default",
                        help="Choose architecture design")
    
    parser.add_argument ("--seed", type=int, default=None,
                        help="Choose architecture design")
    
    hparams = parser.parse_args()
    main(hparams)
