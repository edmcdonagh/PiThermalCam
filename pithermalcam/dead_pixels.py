import numpy as np


def fix_broken_pixels(image):
    dead_pixels = np.argwhere(image == -273.15)
    for px in dead_pixels:
        _fix_dead_pixel_value(px, image)


def _fix_dead_pixel_value(px, image):
    shape = image.shape
    surrounding = image[
        max(0, px[0] - 1) : min(shape[0], px[0] + 2),
        max(0, px[1] - 1) : min(shape[1], px[1] + 2),
    ]
    average = np.sum(surrounding) / (surrounding.size - 1)
    image[px[0], px[1]] = average


def get_min_max(raw_image, exclude_dead_px=True):
    if exclude_dead_px:
        t_min = np.min(raw_image[np.where(raw_image > -50)])
    else:
        t_min = np.min(raw_image)
    t_max = np.max(raw_image)
    return t_min, t_max
