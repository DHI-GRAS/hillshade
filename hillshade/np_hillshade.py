import numpy as np
import tqdm
from typing import Tuple

def elevation(x: np.ndarray, y: np.ndarray, z0: np.ndarray, zenith: float) -> np.ndarray:
    r = (x**2 + y**2)**.5
    return z0 + r * np.tan(zenith)

def check_bounds(x: np.ndarray, y: np.ndarray, bounds: Tuple[int, int]) -> np.ndarray:
    ys = y < bounds[0]
    xs = x < bounds[1]
    yl = y >= 0
    xl = x >= 0
    larger = np.logical_and(xl, yl)
    smaller = np.logical_and(xs, ys)
    return np.logical_and(smaller, larger)

def xyray_vector(azimuth: float) -> np.ndarray:
    """Direction of the sun ray vector in the xy-plane with one component set to one
    depending on the slope of the vector."""
    x, y = np.sin(azimuth), np.cos(azimuth)
    m = y/x
    if m < 1. and m > -1.:
        y = m
        x = 1.
    else:
        y = 1.
        x = 1./m
    return np.array([x, y])

def intersect_shadow(elevation_model: np.ndarray, zenith: float, azimuth: float,
                     dx: int, dy: int) -> np.ndarray:
    """Creates binary mask of shaded regions based on an input elevation model and
    the zenith and azimuth of the sun."""
    azimuth = np.deg2rad(azimuth + 180)
    zenith = np.deg2rad(90 - zenith)
    ray = xyray_vector(azimuth)
    max_elevation = elevation_model.max()

    y0 = np.arange(elevation_model.shape[0])
    x0 = np.arange(elevation_model.shape[1])
    x0, y0 = np.meshgrid(x0, y0)

    true_x = x0.astype(float)
    true_y = y0.astype(float)
    grid_x = x0.astype(int)
    grid_y = y0.astype(int)
    bounds = elevation_model.shape

    intersections = np.zeros(x0.shape)
    within_bounds = check_bounds(true_x, true_y, bounds)
    pbar = tqdm.tqdm()
    while np.any(within_bounds):
        pbar.update()
        z = elevation((grid_x-x0)*dx, (grid_y-y0)*dy, elevation_model, zenith)
        if z.min() > max_elevation:
            break

        # get elevations at xy positions that the rays are currently at
        z_raypos = elevation_model[grid_y.flatten(), grid_x.flatten()].reshape(bounds)
        z_below = z < z_raypos

        inter_mask = np.logical_and(z_below, intersections == 0)
        intersections[inter_mask] = 1

        true_y[within_bounds] += ray[1]
        true_x[within_bounds] += ray[0]
        within_bounds = check_bounds(true_x, true_y, bounds)
        grid_y[within_bounds] = true_y[within_bounds].astype(int)
        grid_x[within_bounds] = true_x[within_bounds].astype(int)
    pbar.close()

    return intersections
