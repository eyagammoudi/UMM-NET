from monai.networks.nets.basic_unet import BasicUNet
import torch.nn as nn 
import torch 
from einops import rearrange

class UNet2D(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = BasicUNet(spatial_dims=2, 
                                in_channels=4, 
                                out_channels=4,
                                features=[16, 16, 32, 64, 128, 16])

    
    def forward(self, x):
        # x: (b, c, d, w, h)
        b, c, d, w, h = x.shape
        x = rearrange(x, "b c d w h -> (b d) c w h")
        x = self.model(x)
        x = rearrange(x, "(b d) c w h -> b c d w h", b=b, c=c)

        return x 
        


