import os
from argparse import ArgumentParser
import errno
import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from .p2model import Challenger
from .utils import set_seed

from test_tube import Experiment, HyperOptArgumentParser
from test_tube.hpc import SlurmCluster

def train(hparams):
    """
    Train model
    
    Parameters
    ----------
    hparams
        Arguments fed into LightningModule
    """
    print("start")
    
    # seed initialization for REPRODUCIBILITY
    if hparams.seed is not None:
        seed = hparams.seed
    else:
        np.random.randint(1, high=10000, size=1)[0]
    print(seed)
    seed_everything(seed)
    np.seterr(divide='ignore', invalid='ignore')
    
    slurm_id = os.environ.get("SLURM_JOBID")
    if slurm_id is None:
        version = None
    else:
        version = str(hparams.design +"_" + slurm_id)
    
    logger = TensorBoardLogger(hparams.logdir,
                               name=hparams.exp_name,
                               version=version)
    checkpoint_path = os.path.join(logger.experiment.get_logdir(),
                                   "checkpoints",
                                   "aaai_{epoch:02d}-{loss:.2e}-{roc_auc:.2f}")
    checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path,
                                          save_top_k=5,
                                          monitor="val/roc_auc",
                                          mode="max")
    model = Challenger(hparams)
    print(model)
    print(hparams)
    trainer = Trainer.from_argparse_args(hparams, 
                                         checkpoint_callback=checkpoint_callback,
                                         logger=logger)
    trainer.fit(model)
    
    

def main(hparams):    
    cluster = SlurmCluster(hyperparam_optimizer=hparams,
                           python_cmd='python3',
                           job_name='test_tube')
    
    cluster.notify_job_status(email='sejin.kim@uhnresearch.ca', on_done=True, on_fail=False)
    cluster.per_experiment_nb_gpus = 1
    cluster.per_experiment_nb_nodes = 1
    
    cluster.memory_mb_per_node = 16384
    cluster.job_time = '2-00:00:00'
    
    cluster.minutes_to_checkpoint_before_walltime = 1
    
    cluster.optimize_parallel_cluster_gpu(train, nb_trials=10, job_name='test_tube', job_display_name='henlo')

    


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = Challenger.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    parser = HyperOptArgumentParser(parents=[parser], strategy='random_search', add_help=False) 

    parser.add_argument("root_directory", type=str,
                        help="Directory containing images and segmentation masks.")
    
    parser.add_argument("clinical_data_path", type=str,
                        help="Path to CSV file containing the clinical data.")
    
    parser.add_argument("--logdir", type=str, default="./data/logs",
                        help="Directory where training logs will be saved.")
    
    parser.add_argument("--cache_dir", type=str, default="./data/data_cache",
                        help=("Directory where the preprocessed data will be saved."))
    
    parser.add_argument("--pred_save_path", type=str, default="./data/predictions/baseline_cnn.csv",
                        help="Directory where final predictions will be saved.")
    
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of worker processes to use for data loading.")
    
    parser.add_argument("--exp_name", type=str, default="challenger",
                        help="Experiment name for logging purposes.")
    
    parser.add_argument ("--design", type=str, default="default",
                        help="Choose architecture design")
                         
    parser.add_argument ("--seed", type=int, default=None,
                        help="Choose architecture design")
    
    
    parser.opt_range('--lr', default=1e-3, type=float, low=1e-5, high=1e-3, tunable=True, log_base=2, nb_samples=8)
    parser.opt_range('--c1', default=1e0, type=float, low=1e-3, high=1e3, tunable=True, log_base=2, nb_samples=8)
    
    hparams = parser.parse_args()
    print(hparams)
    raise ErrorMe

    try:
        main(hparams)
    except OSError as e:
        pass
    
