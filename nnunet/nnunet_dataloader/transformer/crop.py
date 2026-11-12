
import numpy as np
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union
from numpy import random
import torch
from itertools import chain
from .utils import generate_pos_neg_label_crop_centers, \
    create_zero_centered_coordinate_mesh, \
    elastic_deform_coordinates, \
    interpolate_img, scale_coords, \
    augment_gamma, augment_mirroring, is_positive, generate_spatial_bounding_box, \
    Pad


class SpatialCrop:
    """
    General purpose cropper to produce sub-volume region of interest (ROI).
    If a dimension of the expected ROI size is bigger than the input image size, will not crop that dimension.
    So the cropped result may be smaller than the expected ROI, and the cropped results of several images may
    not have exactly the same shape.
    It can support to crop ND spatial (channel-first) data.
    The cropped region can be parameterised in various ways:
        - a list of slices for each spatial dimension (allows for use of -ve indexing and `None`)
        - a spatial center and size
        - the start and end coordinates of the ROI
    """

    def __init__(
            self,
            roi_center: Union[Sequence[int], np.ndarray, None] = None,
            roi_size: Union[Sequence[int], np.ndarray, None] = None,
            roi_start: Union[Sequence[int], np.ndarray, None] = None,
            roi_end: Union[Sequence[int], np.ndarray, None] = None,
    ) -> None:
        """
        Args:
            roi_center: voxel coordinates for center of the crop ROI.
            roi_size: size of the crop ROI, if a dimension of ROI size is bigger than image size,
                will not crop that dimension of the image.
            roi_start: voxel coordinates for start of the crop ROI.
            roi_end: voxel coordinates for end of the crop ROI, if a coordinate is out of image,
                use the end coordinate of image.
            roi_slices: list of slices for each of the spatial dimensions.
        """

        if roi_center is not None and roi_size is not None:
            roi_center = np.asarray(roi_center, dtype=np.int16)
            roi_size = np.asarray(roi_size, dtype=np.int16)
            roi_start_np = np.maximum(roi_center - np.floor_divide(roi_size, 2), 0)
            roi_end_np = np.maximum(roi_start_np + roi_size, roi_start_np)
        else:
            if roi_start is None or roi_end is None:
                raise ValueError("Please specify either roi_center, roi_size or roi_start, roi_end.")
            roi_start_np = np.maximum(np.asarray(roi_start, dtype=np.int16), 0)
            roi_end_np = np.maximum(np.asarray(roi_end, dtype=np.int16), roi_start_np)
        # Allow for 1D by converting back to np.array (since np.maximum will convert to int)
        roi_start_np = roi_start_np if isinstance(roi_start_np, np.ndarray) else np.array([roi_start_np])
        roi_end_np = roi_end_np if isinstance(roi_end_np, np.ndarray) else np.array([roi_end_np])
        # convert to slices
        self.slices = [slice(s, e) for s, e in zip(roi_start_np, roi_end_np)]

    def __call__(self, img: Union[np.ndarray, torch.Tensor]):
        """
        Apply the transform to `img`, assuming `img` is channel-first and
        slicing doesn't apply to the channel dim.
        """
        sd = min(len(self.slices), len(img.shape[1:]))  # spatial dims
        slices = [slice(None)] + self.slices[:sd]
        return img[tuple(slices)]

class CenterSpatialCrop:
    """
    Crop at the center of image with specified ROI size.
    If a dimension of the expected ROI size is bigger than the input image size, will not crop that dimension.
    So the cropped result may be smaller than the expected ROI, and the cropped results of several images may
    not have exactly the same shape.
    Args:
        roi_size: the spatial size of the crop region e.g. [224,224,128]
            if a dimension of ROI size is bigger than image size, will not crop that dimension of the image.
            If its components have non-positive values, the corresponding size of input image will be used.
            for example: if the spatial size of input data is [40, 40, 40] and `roi_size=[32, 64, -1]`,
            the spatial size of output data will be [32, 40, 40].
    """

    def __init__(self, roi_size: Union[Sequence[int], int]) -> None:
        self.roi_size = roi_size

    def __call__(self, img: np.ndarray):
        """
        Apply the transform to `img`, assuming `img` is channel-first and
        slicing doesn't apply to the channel dim.
        """
        assert img.ndim == 4, "img ndim 必须为4， (channel, W, H, D)"
        center = [i // 2 for i in img.shape[1:]]
        cropper = SpatialCrop(roi_center=center, roi_size=self.roi_size)
        return cropper(img)

class CropForegroundImageLabel:
    def __init__(self,
                 select_fn: Callable = is_positive,
                 channel_indices = None,
                 margin = 0,
                 mode = ["constant"]
                 ):
        pass
        self.cropper = CropForeground(
            select_fn=select_fn, channel_indices=channel_indices, margin=margin
        )
        self.mode = mode
    def __call__(self, image, label=None):

        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        box_start, box_end = self.cropper.compute_bounding_box(image)
        print(box_start, box_end)
        # d[self.start_coord_key] = box_start
        # d[self.end_coord_key] = box_end
        # for key, m in self.key_iterator(d, self.mode):
        # self.push_transform(d, key, extra_info={"box_start": box_start, "box_end": box_end})
        image = self.cropper.crop_pad(img=image, box_start=box_start, box_end=box_end, mode=self.mode[0])
        if label is not None :
            if len(label.shape) == 3:
                label = np.expand_dims(label, axis=0)
            label = self.cropper.crop_pad(img=label, box_start=box_start, box_end=box_end, mode=self.mode[1])
            if len(label.shape) == 4:
                label = np.squeeze(label, axis=0)

        return image, label


class CropForeground():
    """
    Crop an image using a bounding box. The bounding box is generated by selecting foreground using select_fn
    at channels channel_indices. margin is added in each spatial dimension of the bounding box.
    The typical usage is to help training and evaluation if the valid part is small in the whole medical image.
    Users can define arbitrary function to select expected foreground from the whole image or specified channels.
    And it can also add margin to every dim of the bounding box of foreground object.
    For example:

    .. code-block:: python

        image = np.array(
            [[[0, 0, 0, 0, 0],
              [0, 1, 2, 1, 0],
              [0, 1, 3, 2, 0],
              [0, 1, 2, 1, 0],
              [0, 0, 0, 0, 0]]])  # 1x5x5, single channel 5x5 image


        def threshold_at_one(x):
            # threshold at 1
            return x > 1


        cropper = CropForeground(select_fn=threshold_at_one, margin=0)
        print(cropper(image))
        [[[2, 1],
          [3, 2],
          [2, 1]]]

    """

    def __init__(
            self,
            select_fn: Callable = is_positive,
            channel_indices = None,
            margin: Union[Sequence[int], int] = 0,
            return_coords: bool = False,
            mode: str = "constant",
            **np_kwargs,
    ) -> None:
        """
        Args:
            select_fn: function to select expected foreground, default is to select values > 0.
            channel_indices: if defined, select foreground only on the specified channels
                of image. if None, select foreground on the whole image.
            margin: add margin value to spatial dims of the bounding box, if only 1 value provided, use it for all dims.
            return_coords: whether return the coordinates of spatial bounding box for foreground.
            k_divisible: make each spatial dimension to be divisible by k, default to 1.
                if `k_divisible` is an int, the same `k` be applied to all the input spatial dimensions.
            mode: available modes for numpy array:{``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
                ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
                available modes for PyTorch Tensor: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
                One of the listed string values or a user supplied function. Defaults to ``"constant"``.
                See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
                https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
            np_kwargs: other args for `np.pad` API, note that `np.pad` treats channel dimension as the first dimension.
                more details: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html

        """
        self.select_fn = select_fn
        self.channel_indices = channel_indices
        self.margin = margin
        self.return_coords = return_coords
        self.mode = mode
        self.np_kwargs = np_kwargs

    def compute_bounding_box(self, img):
        """
        Compute the start points and end points of bounding box to crop.
        And adjust bounding box coords to be divisible by `k`.

        """
        box_start, box_end = generate_spatial_bounding_box(img, self.select_fn, self.channel_indices, self.margin)
        # box_start_, *_ = convert_data_type(box_start, output_type=np.ndarray, dtype=np.int16, wrap_sequence=True)
        # box_end_, *_ = convert_data_type(box_end, output_type=np.ndarray, dtype=np.int16, wrap_sequence=True)
        # print(box_start)
        # print(box_end)
        box_start = np.array(box_start)
        box_end = np.array(box_end)
        orig_spatial_size = box_end - box_start
        # make the spatial size divisible by `k`
        spatial_size = np.array(orig_spatial_size)
        # spatial_size = np.asarray(compute_divisible_spatial_size(orig_spatial_size.tolist(), k=self.k_divisible))
        # update box_start and box_end
        box_start_ = box_start - np.floor_divide(np.asarray(spatial_size) - orig_spatial_size, 2)
        box_end_ = box_start + spatial_size
        return box_start_, box_end_

    def crop_pad(
            self,
            img,
            box_start: np.ndarray,
            box_end: np.ndarray,
            mode = None,
    ):
        """
        Crop and pad based on the bounding box.

        """
        cropped = SpatialCrop(roi_start=box_start, roi_end=box_end)(img)
        pad_to_start = np.maximum(-box_start, 0)
        pad_to_end = np.maximum(box_end - np.asarray(img.shape[1:]), 0)
        pad = list(chain(*zip(pad_to_start.tolist(), pad_to_end.tolist())))
        return BorderPad(spatial_border=pad, mode=mode or self.mode, **self.np_kwargs)(cropped)

    def __call__(self, img, mode = None):
        """
        Apply the transform to `img`, assuming `img` is channel-first and
        slicing doesn't change the channel dim.
        """
        box_start, box_end = self.compute_bounding_box(img)
        cropped = self.crop_pad(img, box_start, box_end, mode)

        if self.return_coords:
            return cropped, box_start, box_end
        return cropped

class BorderPad:
    """
    Pad the input data by adding specified borders to every dimension.

    Args:
        spatial_border: specified size for every spatial border. Any -ve values will be set to 0. It can be 3 shapes:

            - single int number, pad all the borders with the same size.
            - length equals the length of image shape, pad every spatial dimension separately.
              for example, image shape(CHW) is [1, 4, 4], spatial_border is [2, 1],
              pad every border of H dim with 2, pad every border of W dim with 1, result shape is [1, 8, 6].
            - length equals 2 x (length of image shape), pad every border of every dimension separately.
              for example, image shape(CHW) is [1, 4, 4], spatial_border is [1, 2, 3, 4], pad top of H dim with 1,
              pad bottom of H dim with 2, pad left of W dim with 3, pad right of W dim with 4.
              the result shape is [1, 7, 11].
        mode: available modes for numpy array:{``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
            ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
            available modes for PyTorch Tensor: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
            One of the listed string values or a user supplied function. Defaults to ``"constant"``.
            See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
            https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        kwargs: other arguments for the `np.pad` or `torch.pad` function.
            note that `np.pad` treats channel dimension as the first dimension.

    """

    def __init__(
            self,
            spatial_border: Union[Sequence[int], int],
            mode = "constant",
            **kwargs,
    ) -> None:
        self.spatial_border = spatial_border
        self.mode = mode
        self.kwargs = kwargs

    def __call__(
            self, img, mode = None
    ):
        """
        Args:
            img: data to be transformed, assuming `img` is channel-first and
                padding doesn't apply to the channel dim.
            mode: available modes for numpy array:{``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
                ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
                available modes for PyTorch Tensor: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
                One of the listed string values or a user supplied function. Defaults to `self.mode`.
                See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
                https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html

        Raises:
            ValueError: When ``self.spatial_border`` does not contain ints.
            ValueError: When ``self.spatial_border`` length is not one of
                [1, len(spatial_shape), 2*len(spatial_shape)].

        """
        spatial_shape = img.shape[1:]
        spatial_border = self.spatial_border
        if not all(isinstance(b, int) for b in spatial_border):
            raise ValueError(f"self.spatial_border must contain only ints, got {spatial_border}.")
        spatial_border = tuple(max(0, b) for b in spatial_border)

        if len(spatial_border) == 1:
            data_pad_width = [(spatial_border[0], spatial_border[0]) for _ in spatial_shape]
        elif len(spatial_border) == len(spatial_shape):
            data_pad_width = [(sp, sp) for sp in spatial_border[: len(spatial_shape)]]
        elif len(spatial_border) == len(spatial_shape) * 2:
            data_pad_width = [(spatial_border[2 * i], spatial_border[2 * i + 1]) for i in range(len(spatial_shape))]
        else:
            raise ValueError(
                f"Unsupported spatial_border length: {len(spatial_border)}, available options are "
                f"[1, len(spatial_shape)={len(spatial_shape)}, 2*len(spatial_shape)={2*len(spatial_shape)}]."
            )

        all_pad_width = [(0, 0)] + data_pad_width
        padder = Pad(all_pad_width, mode or self.mode, **self.kwargs)
        return padder(img)

class RandCropByPosNegLabel:
    """
    Crop random fixed sized regions with the center being a foreground or background voxel
    based on the Pos Neg Ratio.
    And will return a list of arrays for all the cropped images.
    For example, crop two (3 x 3) arrays from (5 x 5) array with pos/neg=1::
        [[[0, 0, 0, 0, 0],
          [0, 1, 2, 1, 0],            [[0, 1, 2],     [[2, 1, 0],
          [0, 1, 3, 0, 0],     -->     [0, 1, 3],      [3, 0, 0],
          [0, 0, 0, 0, 0],             [0, 0, 0]]      [0, 0, 0]]
          [0, 0, 0, 0, 0]]]
    If a dimension of the expected spatial size is bigger than the input image size,
    will not crop that dimension. So the cropped result may be smaller than expected size, and the cropped
    results of several images may not have exactly same shape.
    Args:
        spatial_size: the spatial size of the crop region e.g. [224, 224, 128].
            if a dimension of ROI size is bigger than image size, will not crop that dimension of the image.
            if its components have non-positive values, the corresponding size of `label` will be used.
            for example: if the spatial size of input data is [40, 40, 40] and `spatial_size=[32, 64, -1]`,
            the spatial size of output data will be [32, 40, 40].
        label: the label image that is used for finding foreground/background, if None, must set at
            `self.__call__`.  Non-zero indicates foreground, zero indicates background.
        pos: used with `neg` together to calculate the ratio ``pos / (pos + neg)`` for the probability
            to pick a foreground voxel as a center rather than a background voxel.
        neg: used with `pos` together to calculate the ratio ``pos / (pos + neg)`` for the probability
            to pick a foreground voxel as a center rather than a background voxel.
        num_samples: number of samples (crop regions) to take in each list.
        image: optional image data to help select valid area, can be same as `img` or another image array.
            if not None, use ``label == 0 & image > image_threshold`` to select the negative
            sample (background) center. So the crop center will only come from the valid image areas.
        image_threshold: if enabled `image`, use ``image > image_threshold`` to determine
            the valid image content areas.
        fg_indices: if provided pre-computed foreground indices of `label`, will ignore above `image` and
            `image_threshold`, and randomly select crop centers based on them, need to provide `fg_indices`
            and `bg_indices` together, expect to be 1 dim array of spatial indices after flattening.
            a typical usage is to call `FgBgToIndices` transform first and cache the results.
        bg_indices: if provided pre-computed background indices of `label`, will ignore above `image` and
            `image_threshold`, and randomly select crop centers based on them, need to provide `fg_indices`
            and `bg_indices` together, expect to be 1 dim array of spatial indices after flattening.
            a typical usage is to call `FgBgToIndices` transform first and cache the results.
    Raises:
        ValueError: When ``pos`` or ``neg`` are negative.
        ValueError: When ``pos=0`` and ``neg=0``. Incompatible values.
    """

    def __init__(
            self,
            spatial_size: Union[Sequence[int], int],
            data_key="data",
            seg_key="seg",
            pos: float = 1.0,
            neg: float = 1.0,
            num_samples: int = 1,
            image_threshold: float = 0.0,
    ) -> None:
        self.data_key = data_key
        self.seg_key = seg_key
        self.spatial_size = spatial_size
        if pos < 0 or neg < 0:
            raise ValueError(f"pos and neg must be nonnegative, got pos={pos} neg={neg}.")
        if pos + neg == 0:
            raise ValueError("Incompatible values: pos=0 and neg=0.")
        self.pos_ratio = pos / (pos + neg)
        self.num_samples = num_samples
        self.image_threshold = image_threshold
        self.centers: Optional[List[List[np.ndarray]]] = None

    def randomize(
            self,
            label: np.ndarray,
            image: Optional[np.ndarray] = None,
    ):
        self.spatial_size = self.spatial_size

        fg_indices_, bg_indices_ = map_binary_to_indices(label, image, self.image_threshold)

        centers = generate_pos_neg_label_crop_centers(
            self.spatial_size, self.num_samples, self.pos_ratio, label.shape[1:], fg_indices_, bg_indices_
        )
        return centers

    def __call__(
            self,
            **data_dict,
    ) :
        """
        Args:
            img: input data to crop samples from based on the pos/neg ratio of `label` and `image`.
                Assumes `img` is a channel-first array.
            label: the label image that is used for finding foreground/background, if None, use `self.label`.
            image: optional image data to help select valid area, can be same as `img` or another image array.
                use ``label == 0 & image > image_threshold`` to select the negative sample(background) center.
                so the crop center will only exist on valid image area. if None, use `self.image`.
            fg_indices: foreground indices to randomly select crop centers,
                need to provide `fg_indices` and `bg_indices` together.
            bg_indices: background indices to randomly select crop centers,
                need to provide `fg_indices` and `bg_indices` together.
        """

        result_image = []
        result_label = []
        for b in range(len(data_dict[self.data_key])):
            label = data_dict[self.seg_key][b]
            if len(label.shape) == 3:
                label = np.expand_dims(label, axis=0)
            image = data_dict[self.data_key][b]

            centers = self.randomize(label, image)

            for center in centers:
                cropper = SpatialCrop(roi_center=tuple(center), roi_size=self.spatial_size)  # type: ignore
                image_crop = cropper(image)
                label_crop = cropper(label)
                if len(label_crop.shape) == 4:
                    label_crop = np.squeeze(label_crop, axis=0)

                result_image.append(image_crop)
                result_label.append(label_crop)

        data_dict[self.data_key] = np.array(result_image)
        data_dict[self.seg_key] = np.array(result_label)

        return data_dict

def map_binary_to_indices(
        label: np.ndarray,
        image: Optional[np.ndarray] = None,
        image_threshold: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the foreground and background of input label data, return the indices after fattening.
    For example:
    ``label = np.array([[[0, 1, 1], [1, 0, 1], [1, 1, 0]]])``
    ``foreground indices = np.array([1, 2, 3, 5, 6, 7])`` and ``background indices = np.array([0, 4, 8])``
    Args:
        label: use the label data to get the foreground/background information.
        image: if image is not None, use ``label = 0 & image > image_threshold``
            to define background. so the output items will not map to all the voxels in the label.
        image_threshold: if enabled `image`, use ``image > image_threshold`` to
            determine the valid image content area and select background only in this area.
    """
    # Prepare fg/bg indices
    if label.shape[0] > 1:
        label = label[1:]  # for One-Hot format data, remove the background channel
    label_flat = np.any(label, axis=0).ravel()  # in case label has multiple dimensions
    fg_indices = np.nonzero(label_flat)[0]
    if image is not None:
        img_flat = np.any(image > image_threshold, axis=0).ravel()
        bg_indices = np.nonzero(np.logical_and(img_flat, ~label_flat))[0]
    else:
        bg_indices = np.nonzero(~label_flat)[0]

    return fg_indices, bg_indices


#####

def get_valid_patch_size(image_size: Sequence[int], patch_size: Union[Sequence[int], int]) -> Tuple[int, ...]:
    """
    Given an image of dimensions `image_size`, return a patch size tuple taking the dimension from `patch_size` if this is
    not 0/None. Otherwise, or if `patch_size` is shorter than `image_size`, the dimension from `image_size` is taken. This ensures
    the returned patch size is within the bounds of `image_size`. If `patch_size` is a single number this is interpreted as a
    patch of the same dimensionality of `image_size` with that size in each dimension.
    """
    ndim = len(image_size)

    # ensure patch size dimensions are not larger than image dimension, if a dimension is None or 0 use whole dimension
    return tuple(min(ms, ps or ms) for ms, ps in zip(image_size, patch_size))
def get_random_patch(
        dims: Sequence[int], patch_size: Sequence[int], rand_state: Optional[np.random.RandomState] = None
) -> Tuple[slice, ...]:
    """
    Returns a tuple of slices to define a random patch in an array of shape `dims` with size `patch_size` or the as
    close to it as possible within the given dimension. It is expected that `patch_size` is a valid patch for a source
    of shape `dims` as returned by `get_valid_patch_size`.
    Args:
        dims: shape of source array
        patch_size: shape of patch size to generate
        rand_state: a random state object to generate random numbers from
    Returns:
        (tuple of slice): a tuple of slice objects defining the patch
    """

    # choose the minimal corner of the patch
    rand_int = np.random.randint if rand_state is None else rand_state.randint
    min_corner = tuple(rand_int(0, ms - ps + 1) if ms > ps else 0 for ms, ps in zip(dims, patch_size))

    # create the slices for each dimension which define the patch in the source array
    return tuple(slice(mc, mc + ps) for mc, ps in zip(min_corner, patch_size))

class RandSpatialCropd:

    def __init__(
            self,
            roi_size: Union[Sequence[int], int],
            data_key="data",
            seg_key="seg",
    ) -> None:
        self.roi_size = roi_size
        self.data_key = data_key
        self.seg_key = seg_key
        self._slices: Optional[Tuple[slice, ...]] = None
        self._size: Optional[Sequence[int]] = None

    def randomize(self, img_size: Sequence[int]) -> None:
        self._size = self.roi_size
        valid_size = get_valid_patch_size(img_size, self._size)
        self._slices = (slice(None),) + get_random_patch(img_size, valid_size)

    def __call__(self, **data_dict):

        result_image = []
        result_label = []
        for b in range(len(data_dict[self.data_key])):
            label = data_dict[self.seg_key][b]

            if len(label.shape) == 3:
                label = np.expand_dims(label, axis=0)
            image = data_dict[self.data_key][b]

            self.randomize(image.shape[1:])  # type: ignore

            image = image[self._slices]
            label = label[self._slices]
            result_image.append(image)

            if label.shape[0] == 1:
                label = label.squeeze(0)

            result_label.append(label)

        data_dict[self.data_key] = np.array(result_image)
        data_dict[self.seg_key] = np.array(result_label)

        return data_dict

