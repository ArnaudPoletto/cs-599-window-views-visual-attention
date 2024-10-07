import numpy as np
from sklearn.neighbors import KernelDensity

def get_kde_density(
    coordinates: np.ndarray,
    kde_width: int,
    kde_height: int,
    kde_bandwidth: float,
    n_samples: int = -1,
    apply_exponential: bool = False,
) -> np.ndarray:
    """
    Fit Kernel Density Estimation to coordinates and return the density map, summed to 1 and either exponential or not.

    Args:
        coordinates (np.ndarray): Coordinates.
        kde_width (int): Width of the KDE map.
        kde_height (int): Height of the KDE map.
        kde_bandwidth (float): Bandwidth of the KDE.
        n_samples (int, optional): Number of samples to use. Defaults to -1.
        apply_exponential (bool, optional): Apply exponential to the density map. Defaults to False.

    Returns:
        np.ndarray: Density map of the coordinates.
    """
    # Filter out coordinates that are out of bounds
    coordinates = coordinates[
        (coordinates[:, 0] >= 0)
        & (coordinates[:, 0] < kde_width)
        & (coordinates[:, 1] >= 0)
        & (coordinates[:, 1] < kde_height)
    ]

    # Sample a subset of coordinates if needed
    if n_samples > 0 and n_samples < len(coordinates):
        coordinates = coordinates[np.random.choice(len(coordinates), n_samples)]

    # Fit Kernel Density Estimation to gaze coordinates
    x_grid, y_grid = np.meshgrid(
        np.linspace(0, kde_width - 1, kde_width),
        np.linspace(0, kde_height - 1, kde_height),
    )
    grid_sample = np.vstack([x_grid.ravel(), y_grid.ravel()]).T
    kde = KernelDensity(bandwidth=kde_bandwidth, kernel="gaussian")
    kde.fit(coordinates)
    kde_scores = kde.score_samples(grid_sample)
    if apply_exponential:
        kde_scores -= np.max(kde_scores) # For numerical stability
        kde_scores = np.exp(kde_scores)
    kde_scores /= np.sum(kde_scores)
    kde_density = kde_scores.reshape(kde_height, kde_width)

    return kde_density