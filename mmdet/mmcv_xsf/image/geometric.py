import numpy as np
import numbers
import cv2
def impad3D(img,
            *,
            shape=None,
            padding=None,
            pad_val=0,
            padding_mode='constant'):
    """Pad the given image to a certain shape or pad on all sides with
    specified padding mode and padding value.

    Args:
        img (ndarray): Image to be padded.
        shape (tuple[int]): Expected padding shape (h, w). Default: None.
        padding (int or tuple[int]): Padding on each border. If a single int is
            provided this is used to pad all borders. If tuple of length 2 is
            provided this is the padding on left/right and top/bottom
            respectively. If a tuple of length 4 is provided this is the
            padding for the left, top, right and bottom borders respectively.
            Default: None. Note that `shape` and `padding` can not be both
            set.
        pad_val (Number | Sequence[Number]): Values to be filled in padding
            areas when padding_mode is 'constant'. Default: 0.
        padding_mode (str): Type of padding. Should be: constant, edge,
            reflect or symmetric. Default: constant.
            - constant: pads with a constant value, this value is specified
              with pad_val.
            - edge: pads with the last value at the edge of the image.
            - reflect: pads with reflection of image without repeating the last
              value on the edge. For example, padding [1, 2, 3, 4] with 2
              elements on both sides in reflect mode will result in
              [3, 2, 1, 2, 3, 4, 3, 2].
            - symmetric: pads with reflection of image repeating the last value
              on the edge. For example, padding [1, 2, 3, 4] with 2 elements on
              both sides in symmetric mode will result in
              [2, 1, 1, 2, 3, 4, 4, 3]

    Returns:
        ndarray: The padded image.
    """

    assert (shape is not None) ^ (padding is not None)
    if shape is not None:
        width = max(shape[2] - img.shape[2], 0)
        height = max(shape[1] - img.shape[1], 0)
        depth = max(shape[0] - img.shape[0], 0)
        padding = (0, 0, 0, width, height, depth)

    # check pad_val
    if isinstance(pad_val, tuple):
        assert len(pad_val) == img.shape[-1]
    elif not isinstance(pad_val, numbers.Number):
        raise TypeError('pad_val must be a int or a tuple. '
                        f'But received {type(pad_val)}')

    # check padding
    if isinstance(padding, tuple) and len(padding) in [3, 6]:
        if len(padding) == 3:
            padding = (padding[0], padding[1], padding[2], padding[0], padding[1], padding[2])
    elif isinstance(padding, numbers.Number):
        padding = (padding, padding, padding, padding, padding, padding)
    else:
        raise ValueError('Padding must be a int or a 3, or 6 element tuple.'
                         f'But received {padding}')

    # check padding mode
    assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

    border_type = {
        'constant': cv2.BORDER_CONSTANT,
        'edge': cv2.BORDER_REPLICATE,
        'reflect': cv2.BORDER_REFLECT_101,
        'symmetric': cv2.BORDER_REFLECT
    }
    # 给图像边界padding常数就行
    img = np.pad(img, ((padding[2], padding[5]), (padding[1], padding[4]), (padding[0], padding[3])), 'constant',
                 constant_values=pad_val)

    # img = cv2.copyMakeBorder(
    #     img,
    #     padding[2],
    #     padding[5],
    #     padding[1],
    #     padding[4],
    #     padding[0],
    #     padding[3],
    #     border_type[padding_mode],
    #     value=pad_val)

    return img


def impad3D_to_multiple(img, divisor, pad_val=0):
    """Pad an image to ensure each edge to be multiple to some number.

    Args:
        img (ndarray): Image to be padded.
        divisor (int): Padded image edges will be multiple to divisor.
        pad_val (Number | Sequence[Number]): Same as :func:`impad`.

    Returns:
        ndarray: The padded image.
    """
    pad_d = int(np.ceil(img.shape[0] / divisor)) * divisor
    pad_h = int(np.ceil(img.shape[1] / divisor)) * divisor
    pad_w = int(np.ceil(img.shape[2] / divisor)) * divisor
    return impad3D(img, shape=(pad_d, pad_h, pad_w), pad_val=pad_val)