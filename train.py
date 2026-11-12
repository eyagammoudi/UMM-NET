import numpy as np
from dataset.brats_data_utils import get_loader_brats
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
from monai.networks.nets.segresnet import SegResNet
from models.transbts.TransBTS_downsample8x_skipconnection import TransBTS
from models.nestedformer.nested_former import NestedFormer
from models.swinunet2d.swinunet import SwinUnet
from einops import rearrange
from monai.networks.nets.vnet import VNet
from models.modelgenesis.unet3d import UNet3DModelGen
from models.transvw.models.ynet3d import UNet3DTransVW
from monai.networks.nets.attentionunet import AttentionUnet
from models.unet_nested.unet_nested_3d import UNet_Nested3D
set_determinism(123)
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3"
data_dir = "/home/xingzhaohu/sharefs/datasets/brats2020/MICCAI_BraTS2020_TrainingData/"
# logdir = "./logs_brats/swinunetr_multi_gpu/"
# logdir = "./logs_brats/unet2d/"
# logdir = "./logs_brats/unet3d/"
# logdir = "./logs_brats/unetr/"
# logdir = "./logs_brats/tranbts/"
# logdir = "./logs_brats/segresnet/"

# logdir = "./logs_brats/transvw"

# logdir = "./logs_brats/swinunet2d"
# logdir = "./logs_brats/vnet"
# logdir = "./logs_brats/modelsgenesis"
# logdir = "./logs_brats/transvw"
# logdir = "./logs_brats/attentionUNet"
logdir = "./logs_brats/unet_plus"

model_save_path = os.path.join(logdir, "model")
max_epoch = 300
batch_size = 1
val_every = 10
num_gpus = 2
device = "cuda:0"

class SwinUNETR2D(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = SwinUNETR([96, 96], 4, 4, spatial_dims=2)
    
    def forward(self, x):
        b, c, d, w, h = x.shape
        x = rearrange(x, "b c d w h -> (b d) c w h")
        # print(x.shape)
        x = self.model(x)
        x = rearrange(x, "(b d) c w h -> b c d w h", b=b, d=d)
        return x 

class BraTSTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/", master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port, training_script)
        self.window_infer = SlidingWindowInferer(roi_size=[96, 96, 96],
                                        sw_batch_size=2,
                                        overlap=0.25)

        # self.model = SwinUNETR([96, 96, 96], 4, 4)
        # self.model = UNet2D()
        # self.model = UNet3D()
        # self.model = UNETR(4, 4, [96, 96, 96])
        # _, model = TransBTS(dataset='brats', _conv_repr=True, _pe_type="learned")

        # self.model = model

        # self.model = SegResNet(3, 16, 4, 4)

        from models.swinunet2d.config import get_config
        config = get_config()
        self.model = SwinUnet(config, img_size=96, in_channels=4, num_classes=4)

        # self.model = SwinUNETR2D()
        # self.model = VNet(3, 4, 4, bias=True)


        # self.model = UNet3DModelGen(4)
        # self.model = UNet3DTransVW(4)
        # weight_dir = "/home/xingzhaohu/jiuding_code/mutual_learning/logs_brats/Genesis_Chest_CT.pt"
        # weight_dir = "/home/xingzhaohu/jiuding_code/mutual_learning/logs_brats/TransVW_chest_ct.pt"
        # checkpoint = torch.load(weight_dir)
        # state_dict = checkpoint['state_dict']
        # unParalled_state_dict = {}
        # for key in state_dict.keys():
        #     if "down_tr64.ops.0.conv1.weight" in key:
        #         state_dict[key] = state_dict[key].repeat(1, 4, 1, 1, 1)

        #     if "out_tr" not in key:
        #         unParalled_state_dict[key.replace("module.", "")] = state_dict[key]

        # self.model.load_state_dict(unParalled_state_dict, strict=False)

        # self.model = AttentionUnet(3, 4, 4, channels=[32, 64, 128, 256], strides=[2, 2, 2, 2])
        
        self.model = UNet_Nested3D(in_channels=4, n_classes=4)
        self.best_mean_dice = 0.0
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-3)

        self.loss_func = nn.CrossEntropyLoss()

    def training_step(self, batch):
        import time 
        image, label = self.get_input(batch)

        pred = self.model(image)
        loss = self.loss_func(pred, label)
        self.log("train_loss", loss, step=self.global_step)
        return loss 

    # for image, label in data_loader:
    def get_input(self, batch):
        image = batch["image"]
        label = batch["label"]

        label[label == 4] = 3
        if len(label.shape) == 5:
            label = label[:, 0]
        label = label.long()
        return image, label 

    def validation_step(self, batch):
        image, label = self.get_input(batch)
       
        output = self.window_infer(image, self.model).argmax(dim=1).cpu().numpy()
        target = label.cpu().numpy()
        o = output > 0; t = target > 0 # ce
        wt = dice(o, t)
        # core
        o = (output == 1) | (output == 3)
        t = (target == 1) | (target == 3)
        tc = dice(o, t)
        # active
        o = (output == 3);t = (target == 3)
        et = dice(o, t)
        
        return [wt, tc, et]

    def validation_end(self, mean_val_outputs):
        wt, tc, et = mean_val_outputs

        self.log("wt", wt, step=self.epoch)
        self.log("tc", tc, step=self.epoch)
        self.log("et", et, step=self.epoch)

        self.log("mean_dice", (wt+tc+et)/3, step=self.epoch)

        mean_dice = (wt + tc + et) / 3
        if mean_dice > self.best_mean_dice:
            self.best_mean_dice = mean_dice
            save_new_model_and_delete_last(self.model, 
                                            os.path.join(model_save_path, 
                                            f"best_model_{mean_dice:.4f}.pt"), 
                                            delete_symbol="best_model")

        save_new_model_and_delete_last(self.model, 
                                        os.path.join(model_save_path, 
                                        f"final_model_{mean_dice:.4f}.pt"), 
                                        delete_symbol="final_model")

        print(f"wt is {wt}, tc is {tc}, et is {et}, mean_dice is {mean_dice}")

if __name__ == "__main__":

    train_ds, val_ds, test_ds = get_loader_brats(data_dir=data_dir, batch_size=batch_size, fold=0)
    
    trainer = BraTSTrainer(env_type="DDP",
                            max_epochs=max_epoch,
                            batch_size=batch_size,
                            device=device,
                            logdir=logdir,
                            val_every=val_every,
                            num_gpus=num_gpus,
                            master_port=17751,
                            training_script=__file__)

    trainer.train(train_dataset=train_ds, val_dataset=val_ds)
