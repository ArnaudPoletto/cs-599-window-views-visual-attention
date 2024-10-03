import cv2
import numpy as np
from sklearn.neighbors import KernelDensity


def get_kde_density(
    coordinates: np.ndarray,
    saliency_width: int,
    saliency_height: int,
    kde_bandwidth: float,
    apply_exponential: bool,
) -> np.ndarray:
    """
    Fit Kernel Density Estimation to gaze coordinates and return the density map.

    Args:
        coordinates (np.ndarray): Gaze coordinates.
        saliency_width (int): Width of the saliency map.
        saliency_height (int): Height of the saliency map.
        kde_bandwidth (float): Bandwidth of the kernel density estimation.
        apply_exponential (bool): Apply exponential to the density map.

    Returns:
        np.ndarray: Density map of the gaze coordinates.
    """
    # Fit Kernel Density Estimation to gaze coordinates
    x_grid, y_grid = np.meshgrid(
        np.linspace(0, saliency_width - 1, saliency_width),
        np.linspace(0, saliency_height - 1, saliency_height),
    )
    grid_sample = np.vstack([x_grid.ravel(), y_grid.ravel()]).T
    kde = KernelDensity(bandwidth=kde_bandwidth, kernel="gaussian")
    kde.fit(coordinates)
    kde_scores = kde.score_samples(grid_sample)
    if apply_exponential:
        kde_scores -= np.max(kde_scores) # For numerical stability
        kde_scores = np.exp(kde_scores)
    kde_density = kde_scores.reshape(saliency_height, saliency_width)

    return kde_density


def get_saliency_map(
    kde_density: np.ndarray,
    frame_width: int,
    frame_height: int,
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
    saliency_map = cv2.resize(saliency_map, (frame_width, frame_height))
