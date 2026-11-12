#
from bis3d_v2.networks.nets.basic_unet_ibsr import BasicUNet
import numpy as np
import torch
import math
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import setproctitle
from 介绍部分图片.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys

from bis3d_v2.networks.nets.segresnet import SegResNet
from sklearn.model_selection import KFold  ## K折交叉验证
# from utils.utils import SoftmaxLoss, sigmoid_rampup, segmenation_metric, infer_25_uncer, infer2d_uncer
from utils.utils import SoftmaxLossHuanhu, segmenation_metric, infer_25_uncer, infer2d_uncer, sigmoid_rampup
import os
import SimpleITK as sitk
from torch.utils.data import DataLoader, random_split, Dataset, ConcatDataset
import h5py
import random
import glob
import copy

## 2.5d 模型改成2d 多通道
torch.backends.cudnn.deterministic = True
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)

model_name = "huanhu_swin_unet_2d"

model_save_dir = "./state_dict/" + model_name + "/"
if not os.path.exists(model_save_dir):
    os.mkdir(model_save_dir)

epochs = 40
batch_size = 1
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print("device is " + str(device))
lr = 0.0001
in_channels = 1
out_channels = 2 # 只分割肿瘤


with open(model_save_dir + "res.txt", "a+") as f:
    f.write("epoch is {}".format(epochs))
    f.write("\n")

data_paths = glob.glob("./data/resize_data/*.h5")
print("数据共：{} 例".format(len(data_paths)))

train_two_fold = []
test_two_fold = []
train_paths = []
test_paths = []
for i in range(78):
    train_paths.append(data_paths[i])
#
for i in range(78, len(data_paths)):
    test_paths.append(data_paths[i])


train_two_fold.append(train_paths)
test_two_fold.append(test_paths)


# ## 第二折
train_paths = []
test_paths = []
for i in range(78, len(data_paths)):
    train_paths.append(data_paths[i])

for i in range(78):
    test_paths.append(data_paths[i])

train_two_fold.append(train_paths)
test_two_fold.append(test_paths)

class SwinUnet(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(SwinUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config

        self.swin_unet = SwinTransformerSys(img_size=config.DATA.IMG_SIZE,
                                            patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                            in_chans=config.MODEL.SWIN.IN_CHANS,
                                            num_classes=self.num_classes,
                                            embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                            depths=config.MODEL.SWIN.DEPTHS,
                                            num_heads=config.MODEL.SWIN.NUM_HEADS,
                                            window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                            qk_scale=config.MODEL.SWIN.QK_SCALE,
                                            drop_rate=config.MODEL.DROP_RATE,
                                            drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                            ape=config.MODEL.SWIN.APE,
                                            patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                            use_checkpoint=config.TRAIN.USE_CHECKPOINT)

    def forward(self, x):
        input = x.squeeze(dim=1)

        input2d = input.permute(1, 0, 2, 3)
        input2d = input2d.repeat(1, 3, 1, 1)
        # print(f"input 2d shape is {input2d.shape}")
        # if x.size()[1] == 1:
        #     x = x.repeat(1,3,1,1)
        logits = self.swin_unet(input2d)
        logits = logits.transpose(1, 0)
        logits = logits.unsqueeze(dim=0)
        return logits

    def load_from(self, config):
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model"  not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.swin_unet.load_state_dict(pretrained_dict,strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.swin_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3-int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k:v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                        del full_dict[k]

            msg = self.swin_unet.load_state_dict(full_dict, strict=False)
            # print(msg)
        else:
            print("none pretrain")



class Dataset3d(Dataset):
    """
    """
    def __init__(
        self,
        paths
    ) -> None:

        super(Dataset3d, self).__init__()
        self.cache_num = len(paths)
        if self.cache_num > 0:
            self._cache_image = [None] * self.cache_num

            self._cache_label = [None] * self.cache_num

            for i in range(self.cache_num):
                self._cache_image[i], self._cache_label[i] = \
                        self._load_cache_item(paths[i])

    def get_labels(self, label):
        labels = np.zeros(label.shape[1:])
        labels[label[0] == 1] = 1
        # labels[label[1] == 1] = 2
        return labels

    def _load_cache_item(self, d_path):
        h5_image = h5py.File(d_path, "r")
        image = h5_image["image"][()]
        label = h5_image["label"][()]
        h5_image.close()
        # print(image.shape)
        image = image[0:1] # 单模态
        # print(image.shape)

        labels = self.get_labels(label)

        return image, labels

    def __getitem__(self, index):
        image = self._cache_image[index]
        image_mean = image.mean()
        image_std = image.std()
        image = (image - image_mean) / image_std

        label = self._cache_label[index]

        return image.astype(np.float32), label.astype(np.float32)

    def __len__(self):
        return len(self._cache_image)

def train(net_3d, train_data_paths, test_data_paths, k_fold):
    train_ds = Dataset3d(train_data_paths)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    optimizer_3d = optim.Adam(net_3d.parameters(), lr=lr, weight_decay=1e-5)

    criterion_3d = nn.CrossEntropyLoss(weight=torch.tensor([1, 5], device=device, dtype=torch.float32))

    val_ds = Dataset3d(test_data_paths)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    best_metric_3d = None
    end_metric_3d = None

    for epoch in range(epochs):

        print("epoch is " + str(epoch))
        net_3d.train()

        epoch_loss_3d = 0.0

        for image, label in tqdm(train_loader, total=len(train_loader)):
            ## image （1， 1, d, w, h）
            ## label: ( 1, d, w, h）
            # for image, label in train_loader:
            optimizer_3d.zero_grad()

            image = nn.functional.interpolate(image, size=(32, 224, 224), mode="trilinear", align_corners=False)
            label = torch.unsqueeze(label, dim=1)
            label = nn.functional.interpolate(label, size=(32, 224, 224), mode="nearest")
            label = torch.squeeze(label, dim=1).long()

            image = image.to(device)
            label = label.to(device)

            pred_3d = net_3d(image)

            hard_loss_3d = criterion_3d(pred_3d, label)
            epoch_loss_3d += hard_loss_3d.item()
            hard_loss_3d.backward()
            optimizer_3d.step()

        metric_3d = []

        net_3d.eval()

        # for image, label in tqdm(val_loader, total=len(val_loader)):
        for image, label in val_loader:
            image = nn.functional.interpolate(image, size=(32, 224, 224), mode="trilinear", align_corners=False)
            label = torch.unsqueeze(label, dim=1)
            label = nn.functional.interpolate(label, size=(32, 224, 224), mode="nearest")
            label = torch.squeeze(label, dim=1).long()

            image = image.to(device)
            label = label.to(device)

            with torch.no_grad():
                pred_3d = net_3d(image)

                metric_3d.append(segmenation_metric(pred_3d, label))

        metric_3d = torch.tensor(metric_3d, dtype=torch.float32)
        metric_3d = metric_3d.mean(dim=0)

        if best_metric_3d is None or (metric_3d[0][0] > best_metric_3d[0][0]):
            best_metric_3d = metric_3d
            torch.save(net_3d.state_dict(), model_save_dir + str(k_fold) + "_3d_" + model_name + "_best.bin")


        torch.save(net_3d.state_dict(), model_save_dir + str(k_fold) + "_3d_" + model_name + ".bin")

        print("metric 3d is " + str(metric_3d))

        end_metric_3d = metric_3d

        print("best metric 3d is " + str(best_metric_3d))

        print("3d loss is " + str(epoch_loss_3d))

    return_res = {

        "metric_3d": end_metric_3d,
        "best_metric_3d": best_metric_3d,
    }
    return return_res


if __name__ == '__main__':

    from 介绍部分图片.config import get_config
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str,
                        default='../data/Synapse/train_npz', help='root dir for data')
    parser.add_argument('--dataset', type=str,
                        default='Synapse', help='experiment_name')
    parser.add_argument('--list_dir', type=str,
                        default='./lists/lists_Synapse', help='list dir')
    parser.add_argument('--num_classes', type=int,
                        default=out_channels, help='output channel of network')
    parser.add_argument('--output_dir', type=str, help='output dir')
    parser.add_argument('--max_iterations', type=int,
                        default=30000, help='maximum epoch number to train')
    parser.add_argument('--max_epochs', type=int,
                        default=150, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int,
                        default=1, help='batch_size per gpu')
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--deterministic', type=int,  default=1,
                        help='whether use deterministic training')
    parser.add_argument('--base_lr', type=float,  default=0.01,
                        help='segmentation network learning rate')
    parser.add_argument('--img_size', type=int,
                        default=224, help='input patch size of network input')
    parser.add_argument('--seed', type=int,
                        default=1234, help='random seed')
    parser.add_argument('--cfg', type=str, default="./介绍部分图片/swin_tiny_patch4_window7_224_lite.yaml", metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    args = parser.parse_args()

    config = get_config(args)
    res_3d = []

    res_3d_best = []

    for i in range(2):
        net_3d = SwinUnet(config, img_size=args.img_size, num_classes=args.num_classes)
        net_3d.load_from(config)
        net_3d.to(device)

        train_paths = train_two_fold[i]
        test_paths = test_two_fold[i]
        res = train(net_3d, train_paths, test_paths, i
                    )

        with open(model_save_dir + "res.txt", "a+") as f:
            f.write(str(res))
            f.write("\n")

        res_3d.append(res["metric_3d"])
        res_3d_best.append(res["best_metric_3d"])

    res_3d = torch.stack(res_3d, dim=0)

    res_3d_best = torch.stack(res_3d_best, dim=0)

    with open(model_save_dir + "res.txt", "a+") as f:
        f.write("res 3d is {} +- {}\n "
                "res 3d best is {} +- {}\n ".format(res_3d.mean(dim=0), res_3d.std(dim=0),
                                                    res_3d_best.mean(dim=0), res_3d_best.std(dim=0),
                                                    ))

        f.write("\n")
