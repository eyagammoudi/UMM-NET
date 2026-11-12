# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from sklearn.model_selection import KFold  ## K折交叉验证

import os
import json
import math
import numpy as np
import torch
from monai import transforms, data
import SimpleITK as sitk
from tqdm import tqdm 
from torch.utils.data import Dataset 
from .multiprocess_loader import get_multiprocess_dataloader
from .transformer import RandomRotate, \
    GammaTransform, RemoveLabelTransform, MirrorTransform, Compose, NumpyToTensor
from .transformer.crop import RandCropByPosNegLabel
from .transformer.noisy_transformers import GaussianNoiseTransform, GaussianBlurTransform
from .transformer.color_transformers import ContrastAugmentationTransform, BrightnessMultiplicativeTransform
from .transformer.resample_transformers import SimulateLowResolutionTransform
from monai import transforms

train_transform = [
    RandomRotate(angle_spectrum=30, execution_probability=0.2),
    RemoveLabelTransform(-1, 0),
    RandCropByPosNegLabel([96, 96, 96], pos=1.0, neg=1.0, num_samples=2),
    GaussianNoiseTransform(noise_variance=(0, 0.1), execution_probability=0.1),
    GaussianBlurTransform(blur_sigma=(0.5, 1), execution_probability=0.1, p_per_channel=0.5, different_sigma_per_channel=True),
    BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), execution_probability=0.15),
    ContrastAugmentationTransform(contrast_range=(0.75, 1.25), execution_probability=0.15),
    SimulateLowResolutionTransform(zoom_range=(0.5, 1.0), per_channel=True, p_per_channel=0.5,
                                   order_downsample=0, order_upsample=3, execution_probability=0.1),
    GammaTransform((0.7, 1.5), True, True, execution_probability=0.1),
    GammaTransform((0.7, 1.5), False, True, execution_probability=0.1),
    # MirrorTransform((0, 1, 2)),
    NumpyToTensor(),
]

train_transform = Compose(train_transform)

val_transform = [
    RemoveLabelTransform(-1, 0),
    NumpyToTensor()
]
val_transform = Compose(val_transform)

class PretrainDataset(Dataset):
    def __init__(self, datalist):
        super().__init__()
        self.datalist = datalist
        
    def read_data(self, data_path):
        
        file_identifizer = data_path.split("/")[-1].split("_")[-1]
        image_paths = [
            os.path.join(data_path, f"BraTS20_Training_{file_identifizer}_t1.nii.gz"),
            os.path.join(data_path, f"BraTS20_Training_{file_identifizer}_flair.nii.gz"),
            os.path.join(data_path, f"BraTS20_Training_{file_identifizer}_t2.nii.gz"),
            os.path.join(data_path, f"BraTS20_Training_{file_identifizer}_t1ce.nii.gz")
        ]
        seg_path = os.path.join(data_path, f"BraTS20_Training_{file_identifizer}_seg.nii.gz")

        image_data = [sitk.GetArrayFromImage(sitk.ReadImage(p)) for p in image_paths]
        seg_data = sitk.GetArrayFromImage(sitk.ReadImage(seg_path))

        image_data = np.array(image_data).astype(np.float32)
        seg_data = np.expand_dims(np.array(seg_data).astype(np.int32), axis=0)

        return image_data, seg_data

    def __getitem__(self, i):
        
        try:
            image_data, seg_data = self.read_data(self.datalist[i])
        except:
            with open("./bugs.txt", "a+") as f:
                f.write(f"数据读取出现问题，{self.datalist[i]}\n")
            if i != len(self.datalist)-1:
                return self.__getitem__(i+1)
            else :
                return self.__getitem__(i-1)
        
        norm = transforms.NormalizeIntensity(nonzero=True, channel_wise=True)

        # norm = Normalization(channel_wise=True)
        image_data = norm(image_data)

        return image_data, seg_data

    def __len__(self):
        return len(self.datalist)


def get_loader_brats(data_dir, fold=0, num_workers=8):

    all_dirs = os.listdir(data_dir)
    all_paths = [os.path.join(data_dir, d) for d in all_dirs]
    # all_paths = glob.glob("/home/xingzhaohu/sharefs/datasets/brats2020/MICCAI_BraTS2020_TrainingData/*")
    # fold_data = get_kfold_data(all_paths, 5)
    size = len(all_paths)
    train_size = int(0.7 * size)
    val_size = int(0.1 * size)
    train_files = all_paths[:train_size]
    val_files = all_paths[train_size:train_size + val_size]
    test_files = all_paths[train_size+val_size:]
    print(f"train is {len(train_files)}, val is {len(val_files)}, test is {len(test_files)}")
    print(f"test_files is {test_files}")

    train_ds = PretrainDataset(train_files)

    train_loader = get_multiprocess_dataloader(train_ds, train_transform=train_transform, batch_size=1, num_process=8)
    
    val_ds = PretrainDataset(val_files)
    
    val_loader = get_multiprocess_dataloader(val_ds, train_transform=val_transform, batch_size=1)

    test_ds = PretrainDataset(test_files)
    test_loader = get_multiprocess_dataloader(test_ds, train_transform=val_transform, batch_size=1)

    loader = [train_loader, val_loader, test_loader]

    return loader
