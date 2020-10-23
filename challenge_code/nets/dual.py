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

class Dual_MTLR (nn.Module):
    def __init__(self, 
                 dense_factor: int = 1, 
                 n_img_dense: int = 1, 
                 n_concat_dense: int = 0, 
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
       
        super(Dual_MTLR, self).__init__()
        
        img_dense_layers      = [[nn.Linear(512, 512 * dense_factor), nn.ReLU(inplace=True)]]
        img_dense_layers.extend([[nn.Linear(512 * dense_factor, 512 * dense_factor), nn.ReLU(inplace=True)] 
                                  for _ in range(n_img_dense - 1)])
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
        
#         if n_concat_dense <= 0:
#             self.mtlr = MTLR(512 * dense_factor + n_clin_vars, 29, num_events=num_events)
#         else: 
        concat_dense_layers = flatten_layers([[nn.Linear(512 * dense_factor + n_clin_vars, 512 * dense_factor + n_clin_vars),
                                               nn.ReLU(inplace=True)] for _ in range(n_concat_dense)])
        self.mtlr = nn.Sequential(*concat_dense_layers,
                                  MTLR(512 * dense_factor + n_clin_vars, 29, num_events=num_events),)

    def forward (self, x):
        img, clin_var = x
        cnn = self.radiomics (img)
        latent_concat = torch.cat ((cnn, clin_var), dim=1)
        # print(img.shape, clin_var.shape, latent_concat.shape)        
        return self.mtlr(latent_concat)
    
class Dual_Lite(nn.Module):
    def __init__ (self):
        super (Dual_Lite, self).__init__()
        
        self.radiomics = nn.Sequential (
            # block 1
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
        )
        
        self.fc = nn.Sequential (
            nn.Linear (512+n_clin_vars, 512), 
            nn.Dropout (0.4),
            nn.Linear (512, 1)
       )
        
    def forward (self, x):
        img, clin_var = x
        cnn = self.radiomics (img) 
        latent_concat = torch.cat ((cnn, clin_var), dim=1) 
        return self.fc (latent_concat)

    
class Dual_Drop (nn.Module):
    def __init__ (self):
        super (Dual_Drop, self).__init__()
        
        self.radiomics = nn.Sequential (
            # block 1
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
            nn.Linear (512, 1024),
        )
        
        self.fc = nn.Sequential (
            nn.Linear (1024+n_clin_vars, 1024), 
            nn.Dropout (0.4),
            nn.Linear (1024, 1)
       )
        
    def forward (self, x):
        img, clin_var = x
        cnn = self.radiomics (img) # returns 512 
        latent_concat = torch.cat ((cnn, clin_var), dim=1) # concatenates tensors sizes of 512 + 32 = 544
        return self.fc (latent_concat)