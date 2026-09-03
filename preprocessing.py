"""
Resampling of binary volumes. OpenCV is imported on use, so the core
pipeline does not depend on it.
"""
import numpy as np


def resize_volume(img, target_size, threshold=0.1):
    """
    Resamples a binary volume to `target_size` with area interpolation and
    re-binarises it.

    Args:
        img (ndarray): 3-D volume indexed (x, y, z); any dtype.
        target_size (tuple): (nx, ny, nz) of the result.
        threshold (float): fraction of vessel occupancy above which a
            resampled voxel counts as vessel. The default 0.1 favours
            keeping thin vessels connected; 0.5 is the unbiased choice.

    Returns:
        ndarray: float32 volume of 0 and 1.
    """
    import cv2

    img = np.asarray(img, dtype=np.float32)
    if img.ndim != 3 or len(target_size) != 3:
        raise ValueError("img must be 3-D and target_size a triple")
    nx, ny, nz = (int(v) for v in target_size)

    plane = np.empty((nx, ny, img.shape[2]), dtype=np.float32)
    for i in range(img.shape[2]):
        plane[:, :, i] = cv2.resize(img[:, :, i], (ny, nx), interpolation=cv2.INTER_AREA)
    plane = (plane > threshold).astype(np.float32)

    out = np.empty((nx, ny, nz), dtype=np.float32)
    for i in range(nx):
        out[i, :, :] = cv2.resize(plane[i], (nz, ny), interpolation=cv2.INTER_AREA)
    return (out > threshold).astype(np.float32)
