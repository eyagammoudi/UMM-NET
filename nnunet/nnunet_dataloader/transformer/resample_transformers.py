
from builtins import range
import numpy as np
from skimage.transform import resize

def uniform(low, high, size=None):
    """
    wrapper for np.random.uniform to allow it to handle low=high
    :param low:
    :param high:
    :return:
    """
    if low == high:
        if size is None:
            return low
        else:
            return np.ones(size) * low
    else:
        return np.random.uniform(low, high, size)


class SimulateLowResolutionTransform:
    """Downsamples each sample (linearly) by a random factor and upsamples to original resolution again
    (nearest neighbor)

    Info:
    * Uses scipy zoom for resampling.
    * Resamples all dimensions (channels, x, y, z) with same downsampling factor (like isotropic=True from
    linear_downsampling_generator_nilearn)

    Args:
        zoom_range: can be either tuple/list/np.ndarray or tuple of tuple. If tuple/list/np.ndarray, then the zoom
        factor will be sampled from zoom_range[0], zoom_range[1] (zoom < 0 = downsampling!). If tuple of tuple then
        each inner tuple will give a sampling interval for each axis (allows for different range of zoom values for
        each axis

        p_per_channel:

        per_channel (bool): whether to draw a new zoom_factor for each channel or keep one for all channels

        channels (list, tuple): if None then all channels can be augmented. If list then only the channel indices can
        be augmented (but may not always be depending on p_per_channel)

        order_downsample:

        order_upsample:
    """

    def __init__(self, data_key="data", zoom_range=(0.5, 1), per_channel=True, p_per_channel=0.5,
                 channels=None, order_downsample=1, order_upsample=3, execution_probability=0.25,
                 ignore_axes=None):

        self.data_key = data_key
        self.order_upsample = order_upsample
        self.order_downsample = order_downsample
        self.channels = channels
        self.per_channel = per_channel
        self.p_per_channel = p_per_channel
        self.execution_probability = execution_probability
        self.zoom_range = zoom_range
        self.ignore_axes = ignore_axes

    def __call__(self, **data_dict):

        for b in range(len(data_dict[self.data_key])):
            if np.random.uniform() < self.execution_probability:
                data_dict[self.data_key][b] = augment_linear_downsampling_scipy(data_dict[self.data_key][b],
                                                                                zoom_range=self.zoom_range,
                                                                                per_channel=self.per_channel,
                                                                                p_per_channel=self.p_per_channel,
                                                                                channels=self.channels,
                                                                                order_downsample=self.order_downsample,
                                                                                order_upsample=self.order_upsample,
                                                                                ignore_axes=self.ignore_axes)
        return data_dict



def augment_linear_downsampling_scipy(data_sample, zoom_range=(0.5, 1), per_channel=True, p_per_channel=1.0,
                                      channels=None, order_downsample=1, order_upsample=0, ignore_axes=None):
    '''
    Downsamples each sample (linearly) by a random factor and upsamples to original resolution again (nearest neighbor)

    Info:
    * Uses scipy zoom for resampling. A bit faster than nilearn.
    * Resamples all dimensions (channels, x, y, z) with same downsampling factor (like isotropic=True from
    linear_downsampling_generator_nilearn)

    Args:
        zoom_range: can be either tuple/list/np.ndarray or tuple of tuple. If tuple/list/np.ndarray, then the zoom
        factor will be sampled from zoom_range[0], zoom_range[1] (zoom < 0 = downsampling!). If tuple of tuple then
        each inner tuple will give a sampling interval for each axis (allows for different range of zoom values for
        each axis

        p_per_channel: probability for downsampling/upsampling a channel

        per_channel (bool): whether to draw a new zoom_factor for each channel or keep one for all channels

        channels (list, tuple): if None then all channels can be augmented. If list then only the channel indices can
        be augmented (but may not always be depending on p_per_channel)

        order_downsample:

        order_upsample:

        ignore_axes: tuple/list

    '''
    if not isinstance(zoom_range, (list, tuple, np.ndarray)):
        zoom_range = [zoom_range]

    shp = np.array(data_sample.shape[1:])
    dim = len(shp)

    if not per_channel:
        if isinstance(zoom_range[0], (tuple, list, np.ndarray)):
            assert len(zoom_range) == dim
            zoom = np.array([uniform(i[0], i[1]) for i in zoom_range])
        else:
            zoom = uniform(zoom_range[0], zoom_range[1])

        target_shape = np.round(shp * zoom).astype(int)

        if ignore_axes is not None:
            for i in ignore_axes:
                target_shape[i] = shp[i]

    if channels is None:
        channels = list(range(data_sample.shape[0]))

    for c in channels:
        if np.random.uniform() < p_per_channel:
            if per_channel:
                if isinstance(zoom_range[0], (tuple, list, np.ndarray)):
                    assert len(zoom_range) == dim
                    zoom = np.array([uniform(i[0], i[1]) for i in zoom_range])
                else:
                    zoom = uniform(zoom_range[0], zoom_range[1])

                target_shape = np.round(shp * zoom).astype(int)
                if ignore_axes is not None:
                    for i in ignore_axes:
                        target_shape[i] = shp[i]

            downsampled = resize(data_sample[c].astype(float), target_shape, order=order_downsample, mode='edge',
                                 anti_aliasing=False)
            data_sample[c] = resize(downsampled, shp, order=order_upsample, mode='edge',
                                    anti_aliasing=False)

    return data_sample

