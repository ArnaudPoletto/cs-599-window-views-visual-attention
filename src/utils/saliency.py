import cv2
import numpy as np
from sklearn.neighbors import KernelDensity


def get_saliency_map(
    kde_density: np.ndarray,
    width: int,
    height: int,
    saliency_colormap: int,
) -> np.ndarray:
    """
    Generate saliency map from the density map.

    Args:
        kde_density (np.ndarray): Density map of the gaze coordinates.
        frame_width (int): Width of the frame.
        frame_height (int): Height of the frame.
        saliency_colormap (int): Colormap for the saliency map.

    Returns:
        np.ndarray: Saliency map.
    """
    saliency_height, saliency_width = kde_density.shape

    if np.unique(kde_density).size == 1:
        saliency_map = np.zeros((saliency_height, saliency_width, 3), dtype=np.uint8)
    else:
        saliency_map = cv2.normalize(kde_density, None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )
        saliency_map = cv2.applyColorMap(saliency_map, saliency_colormap)
    saliency_map = cv2.resize(saliency_map, (width, height))

    return saliency_map
