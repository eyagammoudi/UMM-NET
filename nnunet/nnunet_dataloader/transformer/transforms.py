
import numpy as np
from typing import Callable, Union

import torch
from numpy import random
from scipy.ndimage import rotate
from .utils import augment_gamma, augment_mirroring
from .utils import resample_data_or_seg


RESAMPLING_SEPARATE_Z_ANISO_THRESHOLD = 3

class Compose:

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, **data_dict):
        index = 0
        for t in self.transforms:
            index += 1
            # import time
            # s = time.time()
            data_dict = t(**data_dict)
            # print(data_dict["data"].shape)
            # print(data_dict["seg"].shape)
            # print("~~~~")
            # e = time.time()
            # print(f"trans {index} spend {e - s}")
        return data_dict

def get_do_separate_z(spacing, anisotropy_threshold=RESAMPLING_SEPARATE_Z_ANISO_THRESHOLD):
    do_separate_z = (np.max(spacing) / np.min(spacing)) > anisotropy_threshold
    return do_separate_z


def get_lowres_axis(new_spacing):
    axis = np.where(max(new_spacing) / np.array(new_spacing) == 1)[0]  # find which axis is anisotropic
    return axis

def resample_patient(data, seg, original_spacing, target_spacing, order_data=3, order_seg=0, force_separate_z=False,
                     order_z_data=0, order_z_seg=0,
                     separate_z_anisotropy_threshold=RESAMPLING_SEPARATE_Z_ANISO_THRESHOLD):
    """
    :param data:
    :param seg:
    :param original_spacing:
    :param target_spacing:
    :param order_data:
    :param order_seg:
    :param force_separate_z: if None then we dynamically decide how to resample along z, if True/False then always
    /never resample along z separately
    :param order_z_seg: only applies if do_separate_z is True
    :param order_z_data: only applies if do_separate_z is True
    :param separate_z_anisotropy_threshold: if max_spacing > separate_z_anisotropy_threshold * min_spacing (per axis)
    then resample along lowres axis with order_z_data/order_z_seg instead of order_data/order_seg

    :return:
    """
    assert not ((data is None) and (seg is None))
    
    if data is not None:
        assert len(data.shape) == 4, "data must be c x y z"
    if seg is not None:
        if len(seg.shape) == 3:
            seg = np.expand_dims(seg, axis=0)

        assert len(seg.shape) == 4, "seg must be c x y z"

    if data is not None:
        shape = np.array(data[0].shape)
    else:
        shape = np.array(seg[0].shape)
    
    new_shape = np.round(((np.array(original_spacing) / np.array(target_spacing)).astype(float) * shape)).astype(int)

    if force_separate_z is not None:
        do_separate_z = force_separate_z
        if force_separate_z:
            axis = get_lowres_axis(original_spacing)
        else:
            axis = None
    else:
        if get_do_separate_z(original_spacing, separate_z_anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(original_spacing)
        elif get_do_separate_z(target_spacing, separate_z_anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(target_spacing)
        else:
            do_separate_z = False
            axis = None

    if axis is not None:
        if len(axis) == 3:
            # every axis has the spacing, this should never happen, why is this code here?
            do_separate_z = False
        elif len(axis) == 2:
            # this happens for spacings like (0.24, 1.25, 1.25) for example. In that case we do not want to resample
            # separately in the out of plane axis
            do_separate_z = False
        else:
            pass

    if data is not None:
        data_reshaped = resample_data_or_seg(data, new_shape, False, axis, order_data, do_separate_z,
                                             order_z=order_z_data)
    else:
        data_reshaped = None
    if seg is not None:
        seg_reshaped = resample_data_or_seg(seg, new_shape, True, axis, order_seg, do_separate_z, order_z=order_z_seg)
        if len(seg_reshaped.shape) == 4:
            seg_reshaped = np.squeeze(seg_reshaped, axis=0)
    else:
        seg_reshaped = None
    


    return data_reshaped, seg_reshaped


class RandomRotate:
    """
    Rotate an array by a random degrees from taken from (-angle_spectrum, angle_spectrum) interval.
    Rotation axis is picked at random from the list of provided axes.
    """

    def __init__(self, data_key="data", seg_key="seg", angle_spectrum=30, axes=None, mode='reflect', order=0, execution_probability=0.2):
        if axes is None:
            axes = [[2, 1]] # 这样就是以后两个维度为平面进行旋转。 第一个维度是深度

        self.data_key = data_key
        self.seg_key = seg_key
        self.angle_spectrum = angle_spectrum
        self.axes = axes
        self.execution_probability = execution_probability
        self.mode = mode
        self.order = order

    def __call__(self, **data_dict):
        for b in range(len(data_dict[self.data_key])):
            if np.random.uniform() < self.execution_probability:
                m = data_dict[self.data_key][b]
                label = data_dict[self.seg_key][b]

                axis = self.axes[np.random.randint(len(self.axes))]
                angle = np.random.randint(-self.angle_spectrum, self.angle_spectrum)
                assert m.ndim == 4, "输入必须为3d图像，第一个维度为channel"
                channels = [rotate(m[c], angle, axes=axis, reshape=False, order=self.order, mode=self.mode, cval=-1) for c
                            in range(m.shape[0])]
                m = np.stack(channels, axis=0)

                if label is not None :
                    if label.ndim == 3:
                        label = rotate(label, angle, axes=axis, reshape=False, order=self.order, mode="nearest", cval=-1)
                    elif label.ndim == 4:
                        channels = [rotate(label[c], angle, axes=axis, reshape=False, order=self.order, mode=self.mode, cval=-1) for c
                                    in range(label.shape[0])]
                        label = np.stack(channels, axis=0)

                data_dict[self.data_key][b] = m
                data_dict[self.seg_key][b] = label

        return data_dict


class Standardize:
    """
    Apply Z-score normalization to a given input tensor, i.e. re-scaling the values to be 0-mean and 1-std.
    Mean and std parameter have to be provided explicitly.
    """

    def __init__(self, a_min, a_max, b_min=0, b_max=1, eps=1e-6, clip=True):
        self.a_min = a_min
        self.a_max = a_max
        self.b_min = b_min
        self.b_max = b_max
        self.eps = eps
        self.clip = clip

    def __call__(self, m):
        img = (m - self.a_min) / (self.a_max - self.a_min)

        if self.clip:
            img = np.clip(img, self.b_min, self.b_max)

        return img

class Normalization():
    def __init__(self, channel_wise=False):
        pass
        self.channel_wise = channel_wise

    def __call__(self, m):
        
        assert len(m.shape) == 4, "image shape err"
        if not self.channel_wise:
            m = (m - m.mean()) / m.std()
        else :
            for i, d in enumerate(m):
                
                slices = d != 0
                _sub = d[slices].mean()
                _div = d[slices].std()

                m[i][slices] = (m[i][slices] - _sub) / (_div+1e-8)

        return m


class GammaTransform:
    def __init__(self, gamma_range=(0.5, 2), invert_image=False, per_channel=False, data_key="data",
                 retain_stats: Union[bool, Callable[[], bool]] = False, execution_probability=1):
        self.execution_probability = execution_probability
        self.retain_stats = retain_stats
        self.per_channel = per_channel
        self.data_key = data_key
        self.gamma_range = gamma_range
        self.invert_image = invert_image

    def __call__(self, **data_dict):
        for b in range(len(data_dict[self.data_key])):
            if np.random.uniform() < self.execution_probability:
                data_dict[self.data_key][b] = augment_gamma(data_dict[self.data_key][b], self.gamma_range,
                                                            self.invert_image,
                                                            per_channel=self.per_channel,
                                                            retain_stats=self.retain_stats)
        return data_dict

class MirrorTransform:
    """ Randomly mirrors data along specified axes. Mirroring is evenly distributed. Probability of mirroring along
    each axis is 0.5

    Args:
        axes (tuple of int): axes along which to mirror

    """

    def __init__(self, axes=(0, 1, 2), data_key="data", label_key="seg", p_per_sample=1):
        self.p_per_sample = p_per_sample
        self.data_key = data_key
        self.label_key = label_key
        self.axes = axes
        if max(axes) > 2:
            raise ValueError("MirrorTransform now takes the axes as the spatial dimensions. What previously was "
                             "axes=(2, 3, 4) to mirror along all spatial dimensions of a 5d tensor (b, c, x, y, z) "
                             "is now axes=(0, 1, 2). Please adapt your scripts accordingly.")

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        seg = data_dict.get(self.label_key)

        for b in range(len(data)):
            if np.random.uniform() < self.p_per_sample:
                sample_seg = None
                if seg is not None:
                    sample_seg = seg[b]
                ret_val = augment_mirroring(data[b], sample_seg, axes=self.axes)
                data[b] = ret_val[0]
                if seg is not None:
                    seg[b] = ret_val[1]

        data_dict[self.data_key] = data
        if seg is not None:
            data_dict[self.label_key] = seg

        return data_dict


class RemoveLabelTransform:
    '''
    Replaces all pixels in data_dict[input_key] that have value remove_label with replace_with and saves the result to
    data_dict[output_key]
    '''

    def __init__(self, remove_label, replace_with=0, input_key="seg", output_key="seg"):
        self.output_key = output_key
        self.input_key = input_key
        self.replace_with = replace_with
        self.remove_label = remove_label

    def __call__(self, **data_dict):
        seg = data_dict[self.input_key]
        seg[seg == self.remove_label] = self.replace_with
        data_dict[self.output_key] = seg
        return data_dict

class NumpyToTensor:
    '''
    Replaces all pixels in data_dict[input_key] that have value remove_label with replace_with and saves the result to
    data_dict[output_key]
    '''

    def __init__(self, data_key="data", seg_key="seg"):
        self.data_key = data_key
        self.seg_key = seg_key

    def __call__(self, **data_dict):

        data_dict[self.data_key] = torch.from_numpy(data_dict[self.data_key])
        seg = data_dict.get(self.seg_key)
        if seg is not None :
            data_dict[self.seg_key] = torch.from_numpy(seg)

        return data_dict


# if __name__ == "__main__":
#     print("数据增强函数测试")
#     r = Random(seed=8)
#     print(r.do_transform(0.5))
#     print(r.do_transform(0.5))
#     print(r.do_transform(0.5))
#     print(r.do_transform(0.5))

#     f = RandomFlip(r.R)
#     image = h5py.File("./BAI_YUE_BIN_data.h5", "r")
#     single_model_image = image["image"][:1]
#     label = image["label"][0]
#     print(f"label shape is {label.shape}")
#     print(single_model_image.shape)


#     sd = Standardize(a_min=single_model_image.min(), a_max=single_model_image.max())
#     single_model_image = sd(single_model_image)
#     print("归一化变换")
#     plot_3d(single_model_image)
#     plot_3d_label(label)

#     # print("随机翻转变换")
#     # single_model_image, label = f(single_model_image, label)
#     # plot_3d(single_model_image)
#     # plot_3d_label(label)

#     # print("随机旋转变换")
#     # ro = RandomRotate(random_state=r.R)
#     # single_model_image, label = ro(single_model_image, label)
#     # print(single_model_image.shape)
#     # plot_3d(single_model_image)
#     # plot_3d_label(label)

#     # print("添加高斯噪声")
#     # gn = AdditiveGaussianNoise(r.R)
#     # single_model_image = gn(single_model_image)
#     # plot_3d(single_model_image)

#     print("添加柏松噪声")

#     pn = AdditivePoissonNoise(r.R)
#     single_model_image = pn(single_model_image)
#     plot_3d(single_model_image)

