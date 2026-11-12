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
from light_training.evaluation.metric import dice, hausdorff_distance_95, jaccard, recall, fscore
from models.uent2d import UNet2D
from models.uent3d import UNet3D
from models.uent25d import UNet25D
from monai.networks.nets.segresnet import SegResNet
from models.transbts.TransBTS_downsample8x_skipconnection import TransBTS
from einops import rearrange
set_determinism(123)
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
data_dir = "/home/xingzhaohu/sharefs/datasets/brats2020/MICCAI_BraTS2020_TrainingData/"

max_epoch = 300
batch_size = 2
val_every = 10
num_gpus = 2
device = "cuda:0"
thres = 0.7

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

class FuseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        _, model = TransBTS(dataset='brats', _conv_repr=True, _pe_type="learned")
        self.model3d = model
        self.model25d = UNet25D()
        self.model2d = SwinUNETR2D()

    def forward(self, x, pred_type="all"):
        if pred_type == "all":
            pred_3d = self.model3d(x)
            pred_25d = self.model25d(x)
            pred_2d = self.model2d(x)

            return pred_3d, pred_25d, pred_2d
        
        elif pred_type == "3d":
            return self.model3d(x)
        
        elif pred_type == "25d":
            return self.model25d(x)

        elif pred_type == "2d":
            return self.model2d(x)

class BraTSTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/", master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port, training_script)
        self.window_infer = SlidingWindowInferer(roi_size=[96, 96, 96],
                                        sw_batch_size=2,
                                        overlap=0.5)

        # self.model = SwinUNETR([96, 96, 96], 4, 4)
        # self.model = UNet2D()
        # self.model = UNet3D()
        # self.model = UNet25D()
        # self.model = UNETR(4, 4, [96, 96, 96])

        # _, model = TransBTS(dataset='brats', _conv_repr=True, _pe_type="learned")
        # self.model = model

        self.model = FuseModel()

        # self.model = SegResNet(3, 16, 4, 4)

        self.best_mean_dice = 0.0
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-3)

        self.loss_func = nn.CrossEntropyLoss()

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
       
        output_3d = self.window_infer(image, self.model, pred_type="3d")
        output_25d = self.window_infer(image, self.model, pred_type="25d")
        output_2d = self.window_infer(image, self.model, pred_type="2d")

        # output = (output_3d + output_25d + output_2d).argmax(dim=1).cpu().numpy()
        # output = output_3d.argmax(dim=1).cpu().numpy()
        # output = output_25d.argmax(dim=1).cpu().numpy()
        output = output_3d.argmax(dim=1).cpu().numpy()

        target = label.cpu().numpy()
        o = output > 0; t = target > 0 # ce
        wt = dice(o, t)
        wt_hd = hausdorff_distance_95(o, t)
        wt_recall = recall(o, t)
        wt_f1 = 1 if wt > thres else 0

        wt_f1_9 = 1 if wt > 0.9 else 0

        # core
        o = (output == 1) | (output == 3)
        t = (target == 1) | (target == 3)
        tc = dice(o, t)
        tc_hd = hausdorff_distance_95(o, t)
        tc_recall = recall(o, t)
        tc_f1 = 1 if tc > thres else 0

        tc_f1_9 = 1 if tc > 0.9 else 0

        # active
        o = (output == 3);t = (target == 3)
        et = dice(o, t)
        et_hd = hausdorff_distance_95(o, t)
        et_recall = recall(o, t)
        et_f1 = 1 if et > thres else 0

        et_f1_9 = 1 if et > 0.9 else 0
        if t.sum() == 0:
            et_f1_9 = 1
            et_f1 = 1

        return [wt, tc, et, wt_hd, tc_hd, et_hd, wt_recall, tc_recall, et_recall, wt_f1, tc_f1, et_f1, wt_f1_9, tc_f1_9, et_f1_9]

if __name__ == "__main__":

    train_ds, val_ds, test_ds = get_loader_brats(data_dir=data_dir, batch_size=batch_size, fold=0)
    
    trainer = BraTSTrainer(env_type="pytorch",
                                    max_epochs=max_epoch,
                                    batch_size=batch_size,
                                    device=device,
                                    val_every=val_every,
                                    num_gpus=1,
                                    master_port=17751,
                                    training_script=__file__)

    # logdir = "./logs_brats/swinunetr/model/final_model_0.8455.pt"
    # logdir = "./logs_brats/unet2d/model/final_model_0.8090.pt"
    # logdir = "./logs_brats/unet3d/model/final_model_0.8237.pt"
    # logdir = "./logs_brats/unet25d/model/best_model_0.8389.pt"

    # logdir = "./logs_brats/unetr/model/best_model_0.8545.pt"
    # logdir = "./logs_brats/tranbts/model/best_model_0.8394.pt"
    # logdir = "./logs_brats/unet_mutual_learning_2/model/best_model_0.8571.pt"
    # logdir = "/home/xingzhaohu/jiuding_code/mutual_learning/logs_brats/unet_mutual_learning_2_nosoft_w001_e1000/model/final_model_0.8673.pt"
    # logdir = "./logs_brats/segresnet/"
    # logdir = "/home/xingzhaohu/jiuding_code/mutual_learning/logs_brats/cross_unet_mutual_learning_2/model/final_model_0.8658.pt"
    logdir = "/home/xingzhaohu/jiuding_code/mutual_learning/logs_brats/cross_unet_mutual_learning_2_e1000/model/final_model_0.8779.pt"

    trainer.load_state_dict(logdir)
    v_mean, v_out = trainer.validation_single_gpu(val_dataset=test_ds)

    print(f"v_mean is {v_mean}")
    # print(f"v_out is {v_out}")

    wt_f1 = np.array([d[-6] for d in v_out])
    tc_f1 = np.array([d[-5] for d in v_out])
    et_f1 = np.array([d[-4] for d in v_out])

    wt_f1_9 = np.array([d[-3] for d in v_out])
    tc_f1_9 = np.array([d[-2] for d in v_out])
    et_f1_9 = np.array([d[-1] for d in v_out])

    label = np.ones_like(wt_f1)

    wt_f1 = recall(wt_f1, label)
    tc_f1 = recall(tc_f1, label)
    et_f1 = recall(et_f1, label)

    wt_f1_9 = recall(wt_f1_9, label)
    tc_f1_9 = recall(tc_f1_9, label)
    et_f1_9 = recall(et_f1_9, label)

    print(f"wt_f1 is {wt_f1}, tc_f1 is {tc_f1}, et_f1 is {et_f1}, wt_f1_9 is {wt_f1_9}, tc_f1_9 is {tc_f1_9}, et_f1_9 is {et_f1_9}")