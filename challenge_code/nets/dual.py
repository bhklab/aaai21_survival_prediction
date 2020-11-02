"""
Variations of the default (baseline) model.
"""
import os
import torch
import torch.nn as nn

from torchmtlr import MTLR

def conv_3d_block (in_c, out_c, act='relu', norm='bn', num_groups=8, *args, **kwargs):
    activations = nn.ModuleDict ([
        ['relu', nn.ReLU(inplace=True)],
        ['lrelu', nn.LeakyReLU(0.1, inplace=True)]
    ])
    
    normalizations = nn.ModuleDict ([
        ['bn', nn.BatchNorm3d(out_c)],
        ['gn', nn.GroupNorm(int(out_c/num_groups), out_c)]
    ])
    
    return nn.Sequential(
        nn.Conv3d(in_c, out_c, *args, **kwargs),
        normalizations[norm],
        activations[act]
    )

"""
specify these variables
"""
n_clin_vars = 20

def flatten_layers(arr):
    return [i for sub in arr for i in sub]

class Image_MTLR (nn.Module):
    def __init__(self, 
                 dense_factor: int = 1, 
                 n_dense: int = 1, 
                 dropout: float = 0.333,
                 num_events: int = 1):
        """
        Parameters
        ----------
        dense_factor
            factor multiplying width of dense layer
        n_img_dense
            number of dense layers
        n_concat_dense
            number of dense layers after concat clinical variables to make N-MTLR as per [1]
            
        References
        ----------
        .. [1] https://arxiv.org/abs/1801.05512v1
        """
       
        super(Image_MTLR, self).__init__()
        
        img_dense_layers      = [[nn.Linear(512, 512 * dense_factor),
                                  nn.ReLU(inplace=True), 
                                  nn.Dropout(dropout)]]
        
        img_dense_layers.extend([[nn.Linear(512 * dense_factor, 512 * dense_factor), 
                                  nn.ReLU(inplace=True), 
                                  nn.Dropout(dropout)] for _ in range(n_dense - 1)])
        
        img_dense_layers = flatten_layers(img_dense_layers)
        
        self.radiomics = nn.Sequential (# block 1
                                        conv_3d_block (1, 64, kernel_size=5),
                                        conv_3d_block (64, 128, kernel_size=3),
                                        nn.MaxPool3d(kernel_size=2, stride=2),

                                        # block 2
                                        conv_3d_block (128, 256, kernel_size=3),
                                        conv_3d_block (256, 512, kernel_size=3),
                                        nn.MaxPool3d(kernel_size=2, stride=2),

                                        # global pool
                                        nn.AdaptiveAvgPool3d(1),
    
                                        # linear layers
                                        nn.Flatten(),
                                        *img_dense_layers,)
        

        self.mtlr = MTLR(512 * dense_factor, 29, num_events=num_events)

    def forward (self, x):
        img, clin_var = x
        cnn = self.radiomics (img)     
        return self.mtlr(cnn)

class Dual_MTLR (nn.Module):
    def __init__(self, 
                 dense_factor: int = 1, 
                 n_dense: int = 1, 
                 dropout: float = 0.333,
                 num_events: int = 1):
        """
        Parameters
        ----------
        dense_factor
            factor multiplying width of dense layer
        n_dense
            number of dense layers after concat clinical variables to make N-MTLR as per [1]
            
        References
        ----------
        .. [1] https://arxiv.org/abs/1801.05512v1
        """
       
        super(Dual_MTLR, self).__init__()
        
        self.radiomics = nn.Sequential (# block 1
                                        conv_3d_block (1, 64, kernel_size=5),
                                        conv_3d_block (64, 128, kernel_size=3),
                                        nn.MaxPool3d(kernel_size=2, stride=2),

                                        # block 2
                                        conv_3d_block (128, 256, kernel_size=3),
                                        conv_3d_block (256, 512, kernel_size=3),
                                        nn.MaxPool3d(kernel_size=2, stride=2),

                                        # global pool
                                        nn.AdaptiveAvgPool3d(1),
    
                                        # linear layers
                                        nn.Flatten(),)
        
        if n_dense <= 0:
            self.mtlr = MTLR(512 + n_clin_vars, 29, num_events=num_events)
        else: 
            fc_layers = [[nn.Linear(512 + n_clin_vars, 512 * dense_factor), 
                          nn.ReLU(inplace=True), 
                          nn.Dropout(dropout)]]   
            
            if n_dense > 1:    
                fc_layers.extend([[nn.Linear(512 * dense_factor, 512 * dense_factor),
                                   nn.ReLU(inplace=True),
                                   nn.Dropout(dropout)] for _ in range(n_dense - 1)])
            
            fc_layers = flatten_layers(fc_layers)
            self.mtlr = nn.Sequential(*fc_layers,
                                      MTLR(512 * dense_factor, 29, num_events=num_events),)

    def forward (self, x):
        img, clin_var = x
        cnn = self.radiomics (img)
        latent_concat = torch.cat((cnn, clin_var), dim=1)
        return self.mtlr(latent_concat)
 