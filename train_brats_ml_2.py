
from models.uent25d import UNet25D
from models.uent2d import UNet2D
from models.uent3d import UNet3D
from models.utils import compute_uncer
import numpy as np
from dataset.brats_data_utils import get_loader_brats
import torch 
import torch.nn as nn 
from monai.networks.nets.basic_unet import BasicUNet
from monai.networks.nets.unetr import UNETR
from monai.inferers import SlidingWindowInferer
from light_training.evaluation.metric import dice, hausdorff_distance_95, jaccard, recall, fscore
from light_training.trainer import Trainer
from monai.utils import set_determinism
from models.transbts.TransBTS_downsample8x_skipconnection import TransBTS
from monai.networks.nets.swin_unetr import SwinUNETR
from einops import rearrange
from light_training.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from light_training.utils.files_helper import save_new_model_and_delete_last
set_determinism(123)
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
data_dir = "/home/xingzhaohu/sharefs/datasets/brats2020/MICCAI_BraTS2020_TrainingData/"
device = "cuda:1"
## 去掉了softmax，软标签学习weight为0.01
# logdir = "./logs_brats/unet_mutual_learning_2_nosoft_w001_e1000/"

# logdir = "./logs_brats/unet_mutual_learning_2_no_softloss/"

# logdir = "./logs_brats/cross_unet_mutual_learning_2_e1000_1/"

logdir = "./logs_brats/unet_mutual_learning_e500/"

model_save_path = os.path.join(logdir, "model")
max_epoch = 500
# max_epoch = 300
batch_size = 2
val_every = 5
num_gpus = 2
env = "pytorch"
uncer_threshold = 0.5

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


def compute_metric(output, target):
    o = output > 0; t = target > 0 # ce
    wt_dice = dice(o, t)
    wt_iou = jaccard(o, t)
    wt_recall = recall(o, t)
    # core
    o = (output == 1) | (output == 3)
    t = (target == 1) | (target == 3)
    tc_dice = dice(o, t)
    tc_iou = jaccard(o, t)
    tc_recall = recall(o, t)
    # active
    o = (output == 3);t = (target == 3)
    et_dice = dice(o, t)
    et_iou = jaccard(o, t)
    et_recall = recall(o, t)

    return wt_dice, wt_iou, wt_recall, tc_dice, tc_iou, tc_recall, et_dice, et_iou, et_recall

class FuseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        _, model = TransBTS(dataset='brats', _conv_repr=True, _pe_type="learned")

        self.model3d = model

        self.model25d = UNet25D()
        # self.model2d = UNet2D()
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
                                        overlap=0.25)

        self.model = FuseModel()

        self.best_mean_dice = 0.0
       
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-3)
        self.scheduler = LinearWarmupCosineAnnealingLR(self.optimizer,
                                                  warmup_epochs=50,
                                                  max_epochs=max_epochs)
        self.loss_func = nn.CrossEntropyLoss()
        self.loss_mse = nn.MSELoss(reduction="none")

    def training_step(self, batch):
       
        image, label = self.get_input(batch)

        pred_3d, pred_25d, pred_2d = self.model(image)
 
        hard_loss_3d = self.loss_func(pred_3d, label)
        hard_loss_25d = self.loss_func(pred_25d, label)
        hard_loss_2d = self.loss_func(pred_2d, label)


        # loss = hard_loss_3d + hard_loss_2d + hard_loss_25d
        pred_3d_s = torch.softmax(pred_3d, dim=1) 
        pred_25d_s = torch.softmax(pred_25d, dim=1)
        pred_2d_s = torch.softmax(pred_2d, dim=1)

        # pred_3d_s = pred_3d
        # pred_25d_s = pred_25d
        # pred_2d_s = pred_2d
        
        
        pred_25d_uncer_map = compute_uncer(pred_25d)
        pred_2d_uncer_map = compute_uncer(pred_2d)
        pred_3d_uncer_map = compute_uncer(pred_3d)
        
        pred_25d_uncer_map_bool = pred_25d_uncer_map < uncer_threshold
        pred_2d_uncer_map_bool = pred_2d_uncer_map < uncer_threshold
        pred_3d_uncer_map_bool = pred_3d_uncer_map < uncer_threshold

        soft_loss_25d = (self.loss_mse(pred_3d_s * pred_25d_uncer_map_bool, pred_25d_s.detach() * pred_25d_uncer_map_bool)).mean()
        soft_loss_2d = (self.loss_mse(pred_3d_s * pred_2d_uncer_map_bool, pred_2d_s.detach() * pred_2d_uncer_map_bool)).mean()
        loss_3d = hard_loss_3d + 0.001 * (soft_loss_25d + soft_loss_2d)

        soft_loss_2d = (self.loss_mse(pred_25d_s * pred_2d_uncer_map_bool, pred_2d_s.detach() * pred_2d_uncer_map_bool) ).mean()
        soft_loss_3d = (self.loss_mse(pred_25d_s * pred_3d_uncer_map_bool, pred_3d_s.detach() * pred_3d_uncer_map_bool) ).mean()
        loss_25d = hard_loss_25d + 0.001 * (soft_loss_2d + soft_loss_3d)

        soft_loss_25d = (self.loss_mse(pred_2d_s * pred_25d_uncer_map_bool, pred_25d_s.detach() * pred_25d_uncer_map_bool) ).mean()
        soft_loss_3d = (self.loss_mse(pred_2d_s * pred_3d_uncer_map_bool, pred_3d_s.detach() * pred_3d_uncer_map_bool) ).mean()
        loss_2d = hard_loss_2d + 0.001 * (soft_loss_25d + soft_loss_3d)

       
        loss = loss_2d + loss_3d + loss_25d

        self.log("train_loss", loss, step=self.global_step)

        return loss

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
        fuse_output = (output_3d + output_25d + output_2d) / 3 

        output_3d = output_3d.argmax(dim=1).cpu().numpy()
        output_25d = output_25d.argmax(dim=1).cpu().numpy()
        output_2d = output_2d.argmax(dim=1).cpu().numpy()
        fuse_output = fuse_output.argmax(dim=1).cpu().numpy()

        target = label.cpu().numpy()

        wt_dice_3d, wt_iou, wt_recall, tc_dice_3d, tc_iou, tc_recall, et_dice_3d, et_iou, et_recall = compute_metric(output_3d, target)
        wt_dice_25d, wt_iou, wt_recall, tc_dice_25d, tc_iou, tc_recall, et_dice_25d, et_iou, et_recall = compute_metric(output_25d, target)
        wt_dice_2d, wt_iou, wt_recall, tc_dice_2d, tc_iou, tc_recall, et_dice_2d, et_iou, et_recall = compute_metric(output_2d, target)
        wt_dice_fuse, wt_iou, wt_recall, tc_dice_fuse, tc_iou, tc_recall, et_dice_fuse, et_iou, et_recall = compute_metric(fuse_output, target)
        
        return [wt_dice_fuse, tc_dice_fuse, et_dice_fuse, wt_dice_3d, tc_dice_3d, et_dice_3d, wt_dice_25d, tc_dice_25d, et_dice_25d, wt_dice_2d, tc_dice_2d, et_dice_2d]

    def validation_end(self, mean_val_outputs):
        wt_dice_fuse, tc_dice_fuse, et_dice_fuse, wt_dice_3d, tc_dice_3d, et_dice_3d, wt_dice_25d, tc_dice_25d, et_dice_25d, wt_dice_2d, tc_dice_2d, et_dice_2d = mean_val_outputs

        self.log("wtfuse", wt_dice_fuse, step=self.epoch)
        self.log("tcfuse", tc_dice_fuse, step=self.epoch)
        self.log("etfuse", et_dice_fuse, step=self.epoch)

        self.log("wt3d", wt_dice_3d, step=self.epoch)
        self.log("tc3d", tc_dice_3d, step=self.epoch)
        self.log("et3d", et_dice_3d, step=self.epoch)

        self.log("wt25d", wt_dice_25d, step=self.epoch)
        self.log("tc25d", tc_dice_25d, step=self.epoch)
        self.log("et25d", et_dice_25d, step=self.epoch)

        self.log("wt2d", wt_dice_2d, step=self.epoch)
        self.log("tc2d", tc_dice_2d, step=self.epoch)
        self.log("et2d", et_dice_2d, step=self.epoch)

        self.log("mean_dice_3d", (wt_dice_3d+tc_dice_3d+et_dice_3d)/3, step=self.epoch)
        self.log("mean_dice_25d", (wt_dice_25d+tc_dice_25d+et_dice_25d)/3, step=self.epoch)
        self.log("mean_dice_2d", (wt_dice_2d+tc_dice_2d+et_dice_2d)/3, step=self.epoch)

        mean_dice = (wt_dice_fuse+tc_dice_fuse+et_dice_fuse) / 3
        self.log("mean_dice_fuse", mean_dice, step=self.epoch)
        
        if mean_dice > self.best_mean_dice:
            self.best_mean_dice = mean_dice
            save_new_model_and_delete_last(self.model, 
                                            os.path.join(model_save_path, 
                                            f"best_model_{mean_dice:.4f}.pt"), 
                                            delete_symbol="best_model")

        save_new_model_and_delete_last(self.model, 
                                            os.path.join(model_save_path, 
                                            f"final_model_{mean_dice:.4f}.pt"), 
                                            delete_symbol="final_model_")

        print(f"wt is {wt_dice_3d}, tc is {tc_dice_3d}, et is {et_dice_3d}, mean_dice is {mean_dice}")

if __name__ == "__main__":

    train_ds, val_ds, test_ds = get_loader_brats(data_dir=data_dir, batch_size=batch_size, fold=0)
    
    trainer = BraTSTrainer(env_type=env,
                            max_epochs=max_epoch,
                            batch_size=batch_size,
                            device=device,
                            logdir=logdir,
                            val_every=val_every,
                            num_gpus=num_gpus,
                            master_port=17751,
                            training_script=__file__)

    trainer.train(train_dataset=train_ds, val_dataset=val_ds)

                # 
    ## test image 
    # for data in train_loader:
    #     print(data["image"].shape)
    #     print(data["label"].shape)
    #     import matplotlib.pyplot as plt 
    #     plt.subplot(1, 5, 1)
    #     plt.imshow(data["image"][0, 0, 62], cmap="gray")
    #     plt.subplot(1, 5, 2)
    #     plt.imshow(data["image"][0, 1, 62], cmap="gray")
    #     plt.subplot(1, 5, 3)
    #     plt.imshow(data["image"][0, 2, 62], cmap="gray")
    #     plt.subplot(1, 5, 4)
    #     plt.imshow(data["image"][0, 3, 62], cmap="gray")
        
    #     plt.subplot(1, 5, 5)
    #     plt.imshow(data["label"][0, 0, 62], cmap="gray")
    #     plt.show()
    #     break