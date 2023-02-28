import numpy as np
from numba import jit


@jit('f8(f8, f8, f8)', nopython=True, nogil=True)
def height(xbase, ybase, angle):
    """Calculate the height of a right triangle whose base is defined by a vector in the xy-plane.
    The offset is added to the triangle height.
    """
    rho = (xbase**2 + ybase**2)**.5
    return rho * np.tan(angle)


@jit('boolean(f8, f8, Tuple((i8, i8)))', nopython=True, nogil=True)
def within_bounds(pixel_x, pixel_y, bounds):
    """Check whether x and y pixel coordinates are within bounds"""
    return (pixel_y < bounds[0]) and (pixel_x < bounds[1]) and (pixel_y > 0) and (pixel_x > 0)


@jit('i8[:,:](f4[:,:], Tuple((f8,f8)), f8, Tuple((f8, f8)), i8, i8)', nopython=True, nogil=True)
def hillshade(elevation_model, resolution, zenith, ray, ystart, yend):
    """Calculate a shaded region for elevation_model[ystart:yend] by looping over every
    pixel of the elevation model and tracing the path towards the sun until an obstacle is
    hit or the maximum elevation of the model is reached. The path is defined by a rasterized
    direction of the sun ray in the xy-plane (ray) and the zenith.

    Params:
        elevation_model (np.ndarray):
            Two-dimensional array specifying the elevation at each point of the grid.
        resolution (tuple):
            resolution in meters of the elevation_model
        zenith (float):
            zenith in degrees
        ray (tuple):
            rasterized XY-direction of azimuth
        ystart (int):
            y-chunk starting index
        yend (int):
            y-chunk ending index
    Returns:
        shadow (np.ndarray):
            an array of ones where there is shade and zeros otherwise
    """
    if max(np.abs(np.array(ray))) != 1.:
        raise ValueError("xy-direction is not rasterized.")
    shadow = np.zeros((yend - ystart, elevation_model.shape[1]), dtype=np.int64)
    zenith = np.deg2rad(90 - zenith)
    dx, dy = ray
    xres, yres = resolution
    z_max = elevation_model.max()
    bounds = elevation_model.shape

    for pixel_y in range(ystart, yend):
        for pixel_x in range(elevation_model.shape[1]):

            pixel_z = elevation_model[pixel_y, pixel_x]
            ray_x = float(pixel_x)
            ray_y = float(pixel_y)
            intersection = None

            while within_bounds(ray_x, ray_y, bounds):
                xbase = (ray_x - pixel_x) * xres
                ybase = (ray_y - pixel_y) * yres
                ray_z = height(xbase, ybase, zenith) + pixel_z
                if ray_z > z_max:
                    break
                if ray_z < elevation_model[int(ray_y), int(ray_x)]:
                    intersection = (ray_y, ray_x)
                    break
                ray_x += dx
                ray_y += dy

            if intersection is not None:
                shadow[pixel_y - ystart, pixel_x] = 1
    return shadow
