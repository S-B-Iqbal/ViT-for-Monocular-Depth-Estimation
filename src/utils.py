"""
"""
import numpy as np
import cv2

def read_image(path):
    """Read image and output RGB image (0-1).
    Args:
        path (str): path to file
    Returns:
        array: RGB image (0-1)
    """
    img = cv2.imread(path)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

    return img

def write_depth(path, depth, bits=1):
    """Write depth map to a png file.
    Args:
        path (str): filepath without extension
        depth (array): depth
    """

    depth_min = depth.min()
    depth_max = depth.max()

    max_val = (2**(8*bits))-1

    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        out = np.zeros(depth.shape, dtype=depth.type)

    if bits == 1:
        cv2.imwrite(path + ".png", out.astype("uint8"))
    elif bits == 2:
        cv2.imwrite(path + ".png", out.astype("uint16"))

    return

def align_depth(original,predicted):
    """
    Refer issue: https://github.com/isl-org/MiDaS/issues/171
    """
    if original.shape !=predicted.shape:
        raise ValueError(f"Shape of Original Image = {original.shape} does not align with shape of predicted image: {predicted.shape}")
    x = original.copy().flatten()
    y = predicted.copy().flatten()
    A = np.vstack([x, np.ones(len(x))]).T

    s,t = np.linalg.lstsq(A,y, rcond=None)[0]

    aligned_image = (predicted -t)/s

    return aligned_image