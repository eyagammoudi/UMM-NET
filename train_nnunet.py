
from nnunet_dataloader.brats_ds import get_loader_brats
import numpy as np
import torch 
import torch.nn as nn 
from monai.networks.nets.basic_unet import BasicUNet
from monai.networks.nets.unetr import UNETR
from monai.networks.nets.swin_unetr import SwinUNETR
from monai.inferers import SlidingWindowInferer
from light_training.evaluation.metric import dice
from light_training.trainer import Trainer
from monai.utils import set_determinism
from light_training.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from light_training.utils.files_helper import save_new_model_and_delete_last
from models.uent2d import UNet2D
from models.uent3d import UNet3D
from nnunet.network_architecture.generic_UNet import Generic_UNet
from models.transbts.TransBTS_downsample8x_skipconnection import TransBTS
set_determinism(123)
import os
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = "cuda:0"
save_dir = "./logs_brats_nnunet_new/"
save_name = "nnunet_model_e1000.pt"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, save_name)

# def final_nonlin(x):
#     return x 

# model = Generic_UNet(input_channels=4, base_num_features=30, 
#                     num_classes=4, num_pool=5, conv_op=nn.Conv3d, 
#                     norm_op=nn.BatchNorm3d, dropout_op=nn.Dropout3d,
#                     deep_supervision=False,final_nonlin=final_nonlin,)

 
if __name__ == "__main__":
    data_dir = "/home/xingzhaohu/sharefs/datasets/brats2020/MICCAI_BraTS2020_TrainingData/"
    train_loader, _, _ = get_loader_brats(data_dir)
    loss_func = nn.CrossEntropyLoss()
    model = BasicUNet(spatial_dims=3, 
                    in_channels=4, 
                    out_channels=4,
                    features=[32, 32, 64, 128, 256, 32])

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
    scheduler = LinearWarmupCosineAnnealingLR(optimizer,
                                            warmup_epochs=50,
                                            max_epochs=1000)
    report_loss = 0.0
    index = 0

    for e in range(1000):
        for i in tqdm(range(200), total=200):
            index += 1
            data = next(train_loader)
            image = data["data"].to(device)
            label = data["seg"].to(device).long()
            optimizer.zero_grad()

            pred = model(image)
            label[label == 4] = 3
            loss = loss_func(pred, label)
            report_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()

            if index % 100 == 0:
                print(f"epoch is {e}, loss is {report_loss}")
                report_loss = 0.0 
        
        torch.save(model.state_dict(), save_path)





    

