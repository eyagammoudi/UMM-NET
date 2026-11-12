# Copyright (c) ASU GitHub Project.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
################################################################################


import numpy as np
from torch import nn
import torch
import sys
from models.ynet3d import *

t1 = torch.rand(1, 1, 96, 96, 96)
model = UNet3D(n_class=14)
# Load pre-trained weights
weight_dir = './transvw/TransVW_chest_ct.pt'
checkpoint = torch.load(weight_dir, map_location="cpu")
state_dict = checkpoint['state_dict']
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
delete = [key for key in state_dict if "projection_head" in key]
for key in delete: del state_dict[key]
delete = [key for key in state_dict if "prototypes" in key]
for key in delete: del state_dict[key]
for key in state_dict.keys():
    if key in model.state_dict().keys():
        model.state_dict()[key].copy_(state_dict[key])
        print("Copying {} <---- {}".format(key, key))
    elif key.replace("classficationNet.", "") in model.state_dict().keys():
        model.state_dict()[key.replace("classficationNet.", "")].copy_(state_dict[key])
        print("Copying {} <---- {}".format(key.replace("classficationNet.", ""), key))
    else:
        print("Key {} is not found".format(key))

# checkpoint_en = torch.load("./transvw/TransVW_chest_ct.pt", map_location="cpu")
# for name in model.state_dict().keys():
# 	if name in checkpoint_en['state_dict'].keys():
# 		model.state_dict()[name].copy_(checkpoint_en['state_dict'][name])

out = model(t1)
print(out.shape)