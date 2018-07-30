import numpy as np
from numba import jit
import tqdm

@jit("f8(f8, f8, f8, f8)", nopython=True)
def elevation(x, y, z0, zenith):
    r = (x**2 + y**2)**.5
    return z0 + r * np.tan(zenith)

@jit("Tuple((f8, f8))(f8)", nopython=True)
def xyray_vector(azimuth):
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
    return x, y

@jit("boolean(f8, f8, Tuple((i8, i8)))", nopython=True)
def within_bounds(x, y, bounds):
    return (y < bounds[0]) and (x < bounds[1]) and (y > 0) and (x > 0)

@jit("(f4[:,:], i4[:,:], f8, f8, i8, i8, i8)", nopython=True)
def hillshade(model, shadow, zenith, azimuth, res, ystart, yend):
    azimuth = np.deg2rad(azimuth + 180)
    zenith = np.deg2rad(90 - zenith)
    dx, dy = xyray_vector(azimuth)
    z_max  = model.max()
    bounds = model.shape

    for y0 in range(ystart, yend):
        for x0 in range(model.shape[1]):
            z0 = model[y0, x0]
            x = float(x0)
            y = float(y0)

            intersection = None
            while within_bounds(x, y, bounds):
                z = elevation((x-x0)*res, (y-y0)*res, z0, zenith)
                if z > z_max:
                    break
                if z < model[int(y), int(x)]:
                    intersection = (y, x)
                    break
                x += dx; y += dy

            if intersection is not None:
                shadow[y0, x0] = 1
