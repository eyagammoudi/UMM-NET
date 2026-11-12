import torch.nn as nn 
import torch 
from einops import rearrange

# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Sequence, Union
import copy
import torch
import torch.nn as nn

from monai.networks.blocks import Convolution, UpSample
from monai.networks.layers.factories import Conv, Pool
from monai.utils import deprecated_arg, ensure_tuple_rep

__all__ = ["BasicUnet", "Basicunet", "basicunet", "BasicUNet"]


class TwoConv(nn.Sequential):
    """two convolutions."""

    @deprecated_arg(name="dim", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        dim: Optional[int] = None,
        kernel_size=3,
    ):
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_chns: number of input channels.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            bias: whether to have a bias term in convolution blocks.
            dropout: dropout ratio. Defaults to no dropout.

        .. deprecated:: 0.6.0
            ``dim`` is deprecated, use ``spatial_dims`` instead.
        """
        super().__init__()

        if dim is not None:
            spatial_dims = dim
        padding = 1
        if kernel_size == (1, 3, 3):
            padding = (0, 1, 1)
        conv_0 = Convolution(spatial_dims, in_chns, out_chns, act=act, norm=norm, dropout=dropout, bias=bias, padding=padding, kernel_size=kernel_size)
        conv_1 = Convolution(
            spatial_dims, out_chns, out_chns, act=act, norm=norm, dropout=dropout, bias=bias, padding=padding, kernel_size=kernel_size
        )
        self.add_module("conv_0", conv_0)
        self.add_module("conv_1", conv_1)


class Down(nn.Sequential):
    """maxpooling downsampling and two convolutions."""

    @deprecated_arg(name="dim", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        dim: Optional[int] = None,
        stride=2,
        kernel_size=(3, 3, 3),
    ):
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_chns: number of input channels.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            bias: whether to have a bias term in convolution blocks.
            dropout: dropout ratio. Defaults to no dropout.

        .. deprecated:: 0.6.0
            ``dim`` is deprecated, use ``spatial_dims`` instead.
        """
        super().__init__()
        if dim is not None:
            spatial_dims = dim
        max_pooling = Pool["MAX", spatial_dims](kernel_size=stride)
        convs = TwoConv(spatial_dims, in_chns, out_chns, act, norm, bias, dropout, kernel_size=kernel_size)
        self.add_module("max_pooling", max_pooling)
        self.add_module("convs", convs)


class UpCat(nn.Module):
    """upsampling, concatenation with the encoder feature map, two convolutions"""

    @deprecated_arg(name="dim", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        cat_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        pre_conv: Optional[Union[nn.Module, str]] = "default",
        interp_mode: str = "linear",
        align_corners: Optional[bool] = True,
        halves: bool = True,
        dim: Optional[int] = None,
        stride=2,
        kernel_size=3
    ):
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_chns: number of input channels to be upsampled.
            cat_chns: number of channels from the decoder.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            bias: whether to have a bias term in convolution blocks.
            dropout: dropout ratio. Defaults to no dropout.
            upsample: upsampling mode, available options are
                ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.
            pre_conv: a conv block applied before upsampling.
                Only used in the "nontrainable" or "pixelshuffle" mode.
            interp_mode: {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``}
                Only used in the "nontrainable" mode.
            align_corners: set the align_corners parameter for upsample. Defaults to True.
                Only used in the "nontrainable" mode.
            halves: whether to halve the number of channels during upsampling.
                This parameter does not work on ``nontrainable`` mode if ``pre_conv`` is `None`.

        .. deprecated:: 0.6.0
            ``dim`` is deprecated, use ``spatial_dims`` instead.
        """
        super().__init__()
        if dim is not None:
            spatial_dims = dim
        if upsample == "nontrainable" and pre_conv is None:
            up_chns = in_chns
        else:
            up_chns = in_chns // 2 if halves else in_chns
        self.upsample = UpSample(
            spatial_dims,
            in_chns,
            up_chns,
            stride,
            mode=upsample,
            pre_conv=pre_conv,
            interp_mode=interp_mode,
            align_corners=align_corners,
        )
        self.convs = TwoConv(spatial_dims, cat_chns + up_chns, out_chns, act, norm, bias, dropout, kernel_size=kernel_size)

    def forward(self, x: torch.Tensor, x_e: Optional[torch.Tensor]):
        """

        Args:
            x: features to be upsampled.
            x_e: features from the encoder.
        """
        x_0 = self.upsample(x)

        if x_e is not None:
            # handling spatial shapes due to the 2x maxpooling with odd edge lengths.
            dimensions = len(x.shape) - 2
            sp = [0] * (dimensions * 2)
            for i in range(dimensions):
                if x_e.shape[-i - 1] != x_0.shape[-i - 1]:
                    sp[i * 2 + 1] = 1
            x_0 = torch.nn.functional.pad(x_0, sp, "replicate")
            x = self.convs(torch.cat([x_e, x_0], dim=1))  # input channels: (cat_chns + up_chns)
        else:
            x = self.convs(x_0)

        return x


class BasicUNet(nn.Module):
    @deprecated_arg(
        name="dimensions", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead."
    )
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        features: Sequence[int] = (32, 32, 64, 128, 256, 32),
        act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: Union[str, tuple] = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        dimensions: Optional[int] = None,
    ):
        """
        A UNet implementation with 1D/2D/3D supports.

        Based on:

            Falk et al. "U-Net – Deep Learning for Cell Counting, Detection, and
            Morphometry". Nature Methods 16, 67–70 (2019), DOI:
            http://dx.doi.org/10.1038/s41592-018-0261-2

        Args:
            spatial_dims: number of spatial dimensions. Defaults to 3 for spatial 3D inputs.
            in_channels: number of input channels. Defaults to 1.
            out_channels: number of output channels. Defaults to 2.
            features: six integers as numbers of features.
                Defaults to ``(32, 32, 64, 128, 256, 32)``,

                - the first five values correspond to the five-level encoder feature sizes.
                - the last value corresponds to the feature size after the last upsampling.

            act: activation type and arguments. Defaults to LeakyReLU.
            norm: feature normalization type and arguments. Defaults to instance norm.
            bias: whether to have a bias term in convolution blocks. Defaults to True.
                According to `Performance Tuning Guide <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>`_,
                if a conv layer is directly followed by a batch norm layer, bias should be False.
            dropout: dropout ratio. Defaults to no dropout.
            upsample: upsampling mode, available options are
                ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.

        .. deprecated:: 0.6.0
            ``dimensions`` is deprecated, use ``spatial_dims`` instead.

        Examples::

            # for spatial 2D
            >>> net = BasicUNet(spatial_dims=2, features=(64, 128, 256, 512, 1024, 128))

            # for spatial 2D, with group norm
            >>> net = BasicUNet(spatial_dims=2, features=(64, 128, 256, 512, 1024, 128), norm=("group", {"num_groups": 4}))

            # for spatial 3D
            >>> net = BasicUNet(spatial_dims=3, features=(32, 32, 64, 128, 256, 32))

        See Also

            - :py:class:`monai.networks.nets.DynUNet`
            - :py:class:`monai.networks.nets.UNet`

        """
        super().__init__()
        if dimensions is not None:
            spatial_dims = dimensions

        fea = ensure_tuple_rep(features, 6)
        print(f"BasicUNet features: {fea}.")

        self.conv_0 = TwoConv(spatial_dims, in_channels, features[0], act, norm, bias, dropout)
        self.down_1 = Down(spatial_dims, fea[0], fea[1], act, norm, bias, dropout, stride=(1, 2, 2), kernel_size=(1, 3, 3))
        self.down_2 = Down(spatial_dims, fea[1], fea[2], act, norm, bias, dropout, stride=(1, 2, 2), kernel_size=(1, 3, 3))
        self.down_3 = Down(spatial_dims, fea[2], fea[3], act, norm, bias, dropout, stride=(2, 2, 2))
        self.down_4 = Down(spatial_dims, fea[3], fea[4], act, norm, bias, dropout, stride=(2, 1, 1))

        self.upcat_4 = UpCat(spatial_dims, fea[4], fea[3], fea[3], act, norm, bias, dropout, upsample, stride=(2, 1, 1))
        self.upcat_3 = UpCat(spatial_dims, fea[3], fea[2], fea[2], act, norm, bias, dropout, upsample, stride=(2, 2, 2))
        self.upcat_2 = UpCat(spatial_dims, fea[2], fea[1], fea[1], act, norm, bias, dropout, upsample, stride=(1, 2, 2), kernel_size=(1, 3, 3))
        self.upcat_1 = UpCat(spatial_dims, fea[1], fea[0], fea[5], act, norm, bias, dropout, upsample, halves=False, stride=(1, 2, 2), kernel_size=(1, 3, 3))

        self.final_conv = Conv["conv", spatial_dims](fea[5], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: input should have spatially N dimensions
                ``(Batch, in_channels, dim_0[, dim_1, ..., dim_N])``, N is defined by `dimensions`.
                It is recommended to have ``dim_n % 16 == 0`` to ensure all maxpooling inputs have
                even edge lengths.

        Returns:
            A torch Tensor of "raw" predictions in shape
            ``(Batch, out_channels, dim_0[, dim_1, ..., dim_N])``.
        """
        x0 = self.conv_0(x)

        x1 = self.down_1(x0)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)
        x4 = self.down_4(x3)

        u4 = self.upcat_4(x4, x3)
        u3 = self.upcat_3(u4, x2)
        u2 = self.upcat_2(u3, x1)
        u1 = self.upcat_1(u2, x0)

        logits = self.final_conv(u1)
        return logits


BasicUnet = Basicunet = basicunet = BasicUNet

# def infer_25d(model, x):
#     index = 0
#     device = x.device
#     res_pred = None
#     b, c, d, w, h = x.shape

#     for y in torch.split(x, 1, dim=2):
#         # y = y.squeeze(dim=2) # 去掉deep 维度 在channel维度进行cat
#         if index == 0:
#             y_in = torch.zeros((b, 4, 3, h, w), device=device)
#             y = torch.cat([y, x[:, :, index + 1:index+2]], dim=2)
#             y_in[:, :, 1:] = copy.deepcopy(y)
#             del y
#             y_in = y_in.reshape(b, 12, h, w)

#             pred_25d = model.compute_2d(y_in)
#             res_pred = pred_25d.unsqueeze(2)

#         elif index == d - 1:
#             y_in = torch.zeros((b, 4, 3, h, w), device=device)
#             y = torch.cat([y, x[:, :, index - 1:index]], dim=2)
#             y_in[:, :, :2] = copy.deepcopy(y)
#             del y
#             y_in = y_in.reshape(b, 12, h, w)
#             pred_25d = model.compute_2d(y_in)
#             res_pred = torch.cat([res_pred, pred_25d.unsqueeze(dim=2)], dim=2)


#         else:
#             y = torch.cat([x[:, :, index - 1:index], y, x[:, :, index + 1:index+2]], dim=2)
#             y = y.reshape(b, 12, h, w)
#             pred_25d = model.compute_2d(y)
#             res_pred = torch.cat((res_pred, pred_25d.unsqueeze(2)), dim=2)

#         index += 1
#     return res_pred

def infer_25d(model, x):
    index = 0
    device = x.device
    res_pred = None
    b, c, d, w, h = x.shape
    
    for y in torch.split(x, 1, dim=2):
        # y = y.squeeze(dim=2) # 去掉deep 维度 在channel维度进行cat
        if index == 0:
            y_in = torch.zeros((b, 4, 3, h, w), device=device)
            y = torch.cat([y, x[:, :, index + 1:index+2]], dim=2)
            y_in[:, :, 1:] = copy.deepcopy(y)
            del y
            y_in = y_in.reshape(b, 12, h, w)

            res_pred = y_in.unsqueeze(2)

        elif index == d - 1:
            y_in = torch.zeros((b, 4, 3, h, w), device=device)
            y = torch.cat([y, x[:, :, index - 1:index]], dim=2)
            y_in[:, :, :2] = copy.deepcopy(y)
            del y
            y_in = y_in.reshape(b, 12, h, w)
            res_pred = torch.cat([res_pred, y_in.unsqueeze(dim=2)], dim=2)


        else:
            y = torch.cat([x[:, :, index - 1:index], y, x[:, :, index + 1:index+2]], dim=2)
            y = y.reshape(b, 12, h, w)
            res_pred = torch.cat((res_pred, y.unsqueeze(2)), dim=2)

        index += 1
    
    res_pred = res_pred.reshape(b*d, 12, h, w)
    res_pred = model.compute_2d(res_pred)
    res_pred =res_pred.reshape(b, 4, d, w, h)
    return res_pred

class UNet25D(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = BasicUNet(spatial_dims=3, 
                                in_channels=4, 
                                out_channels=4,
                                features=[16, 16, 32, 64, 128, 16])

    def compute_2d(self, x):
        return self.model(x)
    
    def forward(self, x):
        # x: (b, c, d, w, h)
        # b, c, d, w, h = x.shape
        # return infer_25d(self, x)
        return self.model(x)
        # input2d = x[:, :, 0:2, :, :]
        # single = x[:, :, 0:1, :, :]
        # input2d = torch.cat((input2d, single), 2)
        # for i in range(d - 2):
        #     input2dtmp = x[:, :, i:i + 3, :, :]
        #     input2d = torch.cat((input2d, input2dtmp), 0)
        #     if i == d - 3:
        #         f1 = x[:, :, d - 2: d, :, :]
        #         f2 = x[:, :, d - 1: d, :, :]
        #         ff = torch.cat((f1, f2), 2)
        #         input2d = torch.cat((input2d, ff), 0)
        # x = input2d
        # # x = rearrange(x, "g c k w h -> g (c k) w h")
        # x = self.model(x)
        # x = x[:, :, 1]

        # x = rearrange(x, "(b d) c w h -> b c d w h", b=b, d=d)

        return x 
        


