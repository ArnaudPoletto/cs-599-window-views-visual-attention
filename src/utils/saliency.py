import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def get_saliency_map(
    kde_density: np.ndarray,
    width: int,
    height: int,
    saliency_colormap: int,
) -> np.ndarray:
    """
    Generate saliency map from the density map.

    Args:
        kde_density (np.ndarray): The density map of the gaze coordinates.
        frame_width (int): The width of the frame.
        frame_height (int): The height of the frame.
        saliency_colormap (int): The colormap for the saliency map.

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


def get_saliency_map_difference(
    kde_density1: np.ndarray,
    kde_density2: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """
    Generate saliency map difference between two saliency maps.

    Args:
        kde_density1 (np.ndarray): The first density map of the gaze coordinates.
        kde_density2 (np.ndarray): The second density map of the gaze coordinates.
        width (int): The width of the frame.
        height (int): The height of the frame.

    Returns:
        np.ndarray: Saliency difference map with blue-red colormap.
    """
    # Calculate the difference between video and image saliency maps and normalize
    saliency_difference = kde_density1 - kde_density2
    min_value = np.min(saliency_difference)
    max_value = np.max(saliency_difference)

    # Apply colormap
    colormap = plt.get_cmap("coolwarm")
    norm = mcolors.TwoSlopeNorm(vmin=min_value, vcenter=0, vmax=max_value)
    saliency_difference_map = colormap(norm(saliency_difference))[:, :, :3][:, :, ::-1] # BGR to RGB
    saliency_difference_map = (saliency_difference_map * 255).astype(np.uint8)

    # Resize to match the required dimensions
    saliency_difference_map = cv2.resize(saliency_difference_map, (width, height))

    return saliency_difference_map
