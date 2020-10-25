from math import floor, pi
from argparse import ArgumentParser, Namespace
from copy import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.transforms import Compose

import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, mean_squared_error
from sklearn.model_selection import train_test_split
from lifelines.utils import concordance_index

from .p2dataset import RadcureDataset
from .transforms import *
#from .nets.branched import Default, Default_XL, Default_Pro, Default_Pro_Max, Default_Boost, Default_Extra
from .nets.dual import *

from torchmtlr import mtlr_neg_log_likelihood, mtlr_survival, mtlr_risk


class Challenger(pl.LightningModule):
    """A simple convolutional neural network (CNN) for survival prediction.

    The prognostic task is formulated as binary classification of 2-year
    survival. The architecture is based on [1]_, but with the top
    fully-connected layers removed.

    Notes
    -----
    The model is implemented using `pytorch_lightning`. For an introduction
    to the basic ideas, including the structure of a module, see here:
    <https://pytorch-lightning.readthedocs.io/en/latest/introduction_guide.html>

    References
    ---------
    .. [1] A. Hosny et al., ‘Deep learning for lung cancer prognostication:
       A retrospective multi-cohort radiomics study’, PLOS Medicine, vol. 15,
       no. 11, p. e1002711, Nov. 2018.
    """

    def __init__(self, hparams: Namespace):
        """Initialize the module.

        Parameters
        ----------
        hparams
            `Namespace` object containing the model hyperparameters.
            Should usually be generated automatically by `argparse`.
        """
        super().__init__()

        self.hparams = hparams
        # Default, Default_Air, Default_GN, Default_3X, Default_XL, Default_Pro, Default_Pad, Default_Pro_Max
        self.model = Dual_MTLR(dense_factor=1, 
                               n_img_dense=2,    # default==1
                               n_concat_dense=0, # default==0
                               num_events=2)     # added `cancer_death` in p2dataset.py line #101:102
        
        self.apply (self.init_params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass on a batch of examples.

        Parameters
        ----------
        x
            A batch of examples.

        Returns
        -------
        torch.Tensor
            The predicted logits.
        """
        x = self.model (x)
        #print (x.shape)
        return x

    def init_params(self, m: torch.nn.Module):
        """Initialize the parameters of a module.

        Parameters
        ----------
        m
            The module to initialize.

        Notes
        -----
        Convolutional layer weights are initialized from a normal distribution
        as described in [1]_ in `fan_in` mode. The final layer bias is
        initialized so that the expected predicted probability accounts for
        the class imbalance at initialization.

        References
        ----------
        .. [1] K. He et al. ‘Delving Deep into Rectifiers: Surpassing
           Human-Level Performance on ImageNet Classification’,
           arXiv:1502.01852 [cs], Feb. 2015.
        """

        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, a=.1)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            # initialize the final bias so that the predictied probability at
            # init is equal to the proportion of positive samples
            nn.init.constant_(m.bias, -1.5214691)

    def prepare_data(self):
        """Preprocess the data and create training, validation and test
        datasets.

        This method is called automatically by pytorch-lightning.
        """
        test_transform = Compose([
            Normalize(self.hparams.dataset_mean, self.hparams.dataset_std),
            ToTensor()
        ])
        # apply data augmentation only on training set
        train_transform = Compose([
            RandomInPlaneRotation(pi / 6),
            RandomFlip(0),
            # RandomFlip(1),
            RandomFlip(2),
            Normalize(self.hparams.dataset_mean, self.hparams.dataset_std),
            RandomNoise(.05),
            ToTensor(),
            #lambda x: torch.randn(1, 50, 50, 50)
        ])
        print(self.hparams)
        full_dataset = RadcureDataset(self.hparams.root_directory,
                                      self.hparams.clinical_data_path,
                                      self.hparams.patch_size,
                                      target_col=["target_binary", "survival_time"],
                                      train=True,
                                      transform=train_transform,
                                      cache_dir=self.hparams.cache_dir,
                                      num_workers=self.hparams.num_workers)
#         test_dataset = RadcureDataset("/cluster/projects/radiomics/RADCURE-challenge/data",
#                                       "/cluster/projects/radiomics/RADCURE-challenge/data/test/clinical.csv",
        test_dataset = RadcureDataset(self.hparams.root_directory,
                                      self.hparams.clinical_data_path,
                                      self.hparams.patch_size,
                                      train=True,#set back to False at test phase
                                      transform=test_transform,
                                      cache_dir=self.hparams.cache_dir,
                                      num_workers=self.hparams.num_workers)

        # make sure the validation set is balanced
        val_size = floor(.1 / .7 * len(full_dataset)) # use 10% of all data for validation
        full_indices = range(len(full_dataset))
        full_targets = full_dataset.clinical_data["target_binary"]
        train_indices, val_indices = train_test_split(full_indices, test_size=val_size, stratify=full_targets)
        train_dataset, val_dataset = Subset(full_dataset, train_indices), Subset(full_dataset, val_indices)
        val_dataset.dataset = copy(full_dataset)
        val_dataset.dataset.transform = test_transform
        self.pos_weight = torch.tensor(len(full_targets) / full_targets.sum())

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def on_train_start(self):
        """This method is called automatically by pytorch-lightning."""
        print("Dataset sizes")
        print("=============")
        print(f"training:   {len(self.train_dataset)}")
        print(f"validation: {len(self.val_dataset)}")
        print(f"test:       {len(self.test_dataset)}")

        if self.logger is not None:
            # plot a few example images from the training, validation
            # and test datasets
            datasets = {
                "training": self.train_dataset,
                "validation": self.val_dataset
            }
            if len(self.test_dataset) > 0:
                datasets["test"] = self.test_dataset
            for key, dataset in datasets.items():
                imgs = []
                for i in torch.randint(0, len(dataset), (5,)):
                    img = (dataset[i.item()][0][0][:, 25] - 3.) / 6.
                    imgs.append(img)

                    self.logger.experiment.add_images(key,
                                                      torch.stack(imgs, dim=0),
                                                      dataformats="NCHW")


    def configure_optimizers(self):
        """This method is called automatically by pytorch-lightning."""
        optimizer = Adam(self.parameters(),
                         lr=self.hparams.lr,
                         weight_decay=self.hparams.weight_decay)
        scheduler = {
            "scheduler": MultiStepLR(optimizer, milestones=[60, 160, 360]),
            "monitor": "loss",
        }
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        """Run a single training step on a batch of samples.

        This method is called automatically by pytorch-lightning.
        """
        x, y, labels = batch
        output = self.forward(x).squeeze(1)
        # print (type(output))
        # loss = F.binary_cross_entropy_with_logits(output, y.float(), pos_weight=self.pos_weight)
        loss = mtlr_neg_log_likelihood(output, y.float(), self.model, self.hparams.c1, average=True)
        self.log('training/loss', loss)
        
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        """Run a single validation step on a batch of samples.

        This method is called automatically by pytorch-lightning.
        """
        x, y, labels = batch
        output = self.forward(x).squeeze(1)
        # print (type(output))
        # loss = F.binary_cross_entropy_with_logits(output, y.float(), pos_weight=self.pos_weight)
        loss = mtlr_neg_log_likelihood(output, y.float(), self.model, self.hparams.c1, average=True)
        # pred_prob = torch.sigmoid(output)
        
        return {"loss": loss, "pred_prob": output, "y": y, "labels": labels}

    def validation_epoch_end(self, outputs):
        """Compute performance metrics on the validation dataset.

        This method is called automatically by pytorch-lightning.
        """
        loss        = torch.stack([x["loss"] for x in outputs]).mean()
        pred_prob   = torch.cat([x["pred_prob"] for x in outputs]).cpu()          
        y           = torch.cat([x["y"] for x in outputs]).cpu()
        
        true_binary = torch.cat([x["labels"]["target_binary"] for x in outputs]).cpu()
        true_time   = torch.cat([x["labels"]["time"] for x in outputs]).cpu()
        true_event  = torch.cat([x["labels"]["event"] for x in outputs]).cpu()
        true_cancer = torch.cat([x["labels"]["cancer_death"] for x in outputs]).cpu()
                
        two_year_bin    = np.digitize(2, self.train_dataset.dataset.time_bins)
        survival_event  = mtlr_survival(pred_prob[:,:29])
        survival_cancer = mtlr_survival(pred_prob[:,29:])
        pred_event      = 1 - survival_event[:, two_year_bin]
        pred_cancer     = 1 - survival_cancer[:, two_year_bin]
        
        roc_auc_event   = roc_auc_score(true_event, pred_event)
        roc_auc_cancer  = roc_auc_score(true_cancer, pred_cancer)
        avg_prec_event  = average_precision_score(true_event, pred_event)
        avg_prec_cancer = average_precision_score(true_cancer, pred_cancer)
        #print(roc_auc_event, roc_auc_cancer, avg_prec_event, avg_prec_cancer)
        
        pred_risk = mtlr_risk(pred_prob, 2).numpy()
        #print(pred_risk)
        
        ci_event  = concordance_index(true_time, -pred_risk[:, 0], event_observed=true_cancer)
        ci_cancer = concordance_index(true_time, -pred_risk[:, 1], event_observed=true_cancer)
        
        try:
            roc_auc_total = roc_auc_score(y, pred_prob, average='samples')
        except ValueError as e:
            roc_auc_total = float("nan")
        
        avg_prec_total = average_precision_score(y, pred_prob, average='samples')
        # mse_time = mean_squared_error(y, pred_prob)

        # log loss and metrics to Tensorboard
        log = {"val/loss": loss,
               "val/roc_auc": roc_auc_event,
               "val/roc_auc_cancer": roc_auc_cancer,
               "val/precision": avg_prec_event,
               "val/precision_cancer": avg_prec_cancer,
               "val/ci": ci_event,
               "val/ci_cancer": ci_cancer,
               # "val/mse_time": mse_time
               }
        
        self.log_dict(log)
        return {"loss": loss, "roc_auc": roc_auc_total}

    def test_step(self, batch, batch_idx):
        """Run a single test step on a batch of samples.

        This method is called automatically by pytorch-lightning.
        """
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        """Compute performance metrics on the test dataset.

        This method is called automatically by pytorch-lightning.
        """
        pred_prob = torch.cat([x["pred_prob"] for x in outputs]).detach().cpu().numpy()
        y = torch.cat([x["y"] for x in outputs]).detach().cpu().numpy()
        
        try:
            roc_auc = roc_auc_score(y, pred_prob)
        except ValueError:
            roc_auc = float("nan")
            
        avg_prec = average_precision_score(y, pred_prob)
        ids = self.test_dataset.clinical_data["Study ID"]
        np.save(self.hparams.pred_save_path, pred_prob)
        np.save(self.hparams.pred_save_path+"idx.npy", np.array(ids))
        test = np.append(np.copy(pred_prob), np.array(ids), axis=1)
        np.save(self.hparams.pred_save_path+"test.npy", test)
        pd.Series(pred_prob, index=ids, name="binary").to_csv(self.hparams.pred_save_path)
        
        return {
            "roc_auc": roc_auc,
            "average_precision": avg_prec
        }

    def train_dataloader(self):
        """This method is called automatically by pytorch-lightning."""
        return DataLoader(self.train_dataset,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          shuffle=True)
    
    def val_dataloader(self):
        """This method is called automatically by pytorch-lightning."""
        return DataLoader(self.val_dataset,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          shuffle=False)

    def test_dataloader(self):
        """This method is called automatically by pytorch-lightning."""
        return DataLoader(self.test_dataset,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          shuffle=False)
    
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        """Add model-specific hyperparameters to the parent parser."""
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        
        parser.add_argument("--batch_size", type=int, default=16, 
                            help="The batch size.")
        
        parser.add_argument("--lr", type=float, default=3e-4, 
                            help="The initial learning rate.")
        
        parser.add_argument("--weight_decay", type=float, default=1e-5, 
                            help="The amount of weight decay to use.")
        
        parser.add_argument("--patch_size", type=int, default=50, 
                            help="Size of the image patch extracted around each tumour.")
        
        parser.add_argument("--dataset_mean", type=float, default=7.8577905, 
                            help="The mean pixel intensity used for input normalization.")
        
        parser.add_argument("--dataset_std", type=float, default=257.45108, 
                            help="The standard deviation of  pixel intensity used for input normalization.")
        
        parser.add_argument("--c1", type=float, default=1., 
                            help="Regularization term for MTLR backprop.")
        
        
        return parser
