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
from models.uent25d import UNet25D
# from models.uent2d import UNet2D
set_determinism(123)
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"
data_dir = "/home/xingzhaohu/sharefs/datasets/brats2020/MICCAI_BraTS2020_TrainingData/"
logdir = "./logs_brats/unet25d/"
model_save_path = os.path.join(logdir, "model")
max_epoch = 300
batch_size = 2
val_every = 10
num_gpus = 2

class BraTSTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/", master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port, training_script)
        self.window_infer = SlidingWindowInferer(roi_size=[96, 96, 96],
                                        sw_batch_size=2,
                                        overlap=0.25)

        self.model = UNet25D()

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

        # output = self.window_infer(image, self.model).argmax(dim=1).cpu().numpy()
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
    
    trainer = BraTSTrainer(env_type="pytorch",
                                    max_epochs=max_epoch,
                                    batch_size=batch_size,
                                    device="cuda:0",
                                    logdir=logdir,
                                    val_every=val_every,
                                    num_gpus=num_gpus,
                                    master_port=17751,
                                    training_script=__file__)

    trainer.train(train_dataset=train_ds, val_dataset=val_ds)
