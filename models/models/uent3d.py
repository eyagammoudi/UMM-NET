from monai.networks.nets.basic_unet import BasicUNet
import torch.nn as nn 
import torch 

class UNet3D(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = BasicUNet(spatial_dims=3, 
                                in_channels=4, 
                                out_channels=4,
                                features=[16, 16, 32, 64, 128, 16])

    
    def forward(self, x):
        return self.model(x)
        


