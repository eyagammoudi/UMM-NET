
import random 
import numpy as np 
from typing import Tuple, List, Hashable
from scipy.ndimage import gaussian_filter


class GaussianNoiseTransform:
    def __init__(self,data_key="data", noise_variance=(0, 0.1), p_per_channel: float = 1,
                 per_channel: bool = False, execution_probability=0.2):

        self.data_key = data_key
        self.noise_variance = noise_variance
        self.p_per_channel = p_per_channel
        self.per_channel = per_channel
        self.execution_probability = execution_probability

    def __call__(self, **data_dict):

        for b in range(len(data_dict[self.data_key])):
            if np.random.uniform() < self.execution_probability:
                data_dict[self.data_key][b] = augment_gaussian_noise(data_dict[self.data_key][b], self.noise_variance,
                                                                     self.p_per_channel, self.per_channel)
        return data_dict

def augment_gaussian_noise(data_sample: np.ndarray, noise_variance: Tuple[float, float] = (0, 0.1),
                           p_per_channel: float = 1, per_channel: bool = False) -> np.ndarray:
    if not per_channel:
        variance = noise_variance[0] if noise_variance[0] == noise_variance[1] else \
            random.uniform(noise_variance[0], noise_variance[1])
    else:
        variance = None
    for c in range(data_sample.shape[0]):
        if np.random.uniform() < p_per_channel:
            # lol good luck reading this
            variance_here = variance if variance is not None else \
                noise_variance[0] if noise_variance[0] == noise_variance[1] else \
                    random.uniform(noise_variance[0], noise_variance[1])
            # bug fixed: https://github.com/MIC-DKFZ/batchgenerators/issues/86
            data_sample[c] = data_sample[c] + np.random.normal(0.0, variance_here, size=data_sample[c].shape)
    return data_sample

def augment_gaussian_blur(data_sample: np.ndarray, sigma_range: Tuple[float, float], per_channel: bool = True,
                          p_per_channel: float = 1, different_sigma_per_axis: bool = False,
                          p_isotropic: float = 0) -> np.ndarray:
    if not per_channel:
        # Godzilla Had a Stroke Trying to Read This and F***ing Died
        # https://i.kym-cdn.com/entries/icons/original/000/034/623/Untitled-3.png
        sigma = get_range_val(sigma_range) if ((not different_sigma_per_axis) or
                                               ((np.random.uniform() < p_isotropic) and
                                                different_sigma_per_axis)) \
            else [get_range_val(sigma_range) for _ in data_sample.shape[1:]]
    else:
        sigma = None
    for c in range(data_sample.shape[0]):
        if np.random.uniform() <= p_per_channel:
            if per_channel:
                sigma = get_range_val(sigma_range) if ((not different_sigma_per_axis) or
                                                       ((np.random.uniform() < p_isotropic) and
                                                        different_sigma_per_axis)) \
                    else [get_range_val(sigma_range) for _ in data_sample.shape[1:]]
            data_sample[c] = gaussian_filter(data_sample[c], sigma, order=0)
    return data_sample


class GaussianBlurTransform:
    def __init__(self,blur_sigma: Tuple[float, float] = (1, 5), different_sigma_per_channel: bool = True,
                 different_sigma_per_axis: bool = False, p_isotropic: float = 0, p_per_channel: float = 0.5,
                    data_key: str = "data", execution_probability=0.2):
        """

        :param blur_sigma:
        :param data_key:
        :param different_sigma_per_axis: if True, anisotropic kernels are possible
        :param p_isotropic: only applies if different_sigma_per_axis=True, p_isotropic is the proportion of isotropic
        kernels, the rest gets random sigma per axis
        :param different_sigma_per_channel: whether to sample a sigma for each channel or all channels at once
        :param p_per_channel: probability of applying gaussian blur for each channel. Default = 1 (all channels are
        blurred with prob 1)
        """
        self.different_sigma_per_channel = different_sigma_per_channel
        self.p_per_channel = p_per_channel
        self.data_key = data_key
        self.blur_sigma = blur_sigma
        self.different_sigma_per_axis = different_sigma_per_axis
        self.p_isotropic = p_isotropic
        self.execution_probability = execution_probability

    def __call__(self, **data_dict):
        for b in range(len(data_dict[self.data_key])):
            if np.random.uniform() < self.execution_probability:
                data_dict[self.data_key][b] = augment_gaussian_blur(data_dict[self.data_key][b], self.blur_sigma,
                                                                    self.different_sigma_per_channel,
                                                                    self.p_per_channel,
                                                                    different_sigma_per_axis=self.different_sigma_per_axis,
                                                                    p_isotropic=self.p_isotropic)
        return data_dict


def get_range_val(value, rnd_type="uniform"):
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 2:
            if value[0] == value[1]:
                n_val = value[0]
            else:
                orig_type = type(value[0])
                if rnd_type == "uniform":
                    n_val = random.uniform(value[0], value[1])
                elif rnd_type == "normal":
                    n_val = random.normalvariate(value[0], value[1])
                n_val = orig_type(n_val)
        elif len(value) == 1:
            n_val = value[0]
        else:
            raise RuntimeError("value must be either a single value or a list/tuple of len 2")
        return n_val
    else:
        return value