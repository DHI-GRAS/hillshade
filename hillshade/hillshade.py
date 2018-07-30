import numpy as np
from numba import jit


@jit("f8(f8, f8, f8, f8)", nopython=True, nogil=True)
def elevation(x, y, z0, zenith):
    r = (x**2 + y**2)**.5
    return z0 + r * np.tan(zenith)


@jit("boolean(f8, f8, Tuple((i8, i8)))", nopython=True, nogil=True)
def within_bounds(x, y, bounds):
    return (y < bounds[0]) and (x < bounds[1]) and (y > 0) and (x > 0)


@jit("i8[:,:](f4[:,:], f8, Tuple((f8,f8)), Tuple((f8, f8)), i8, i8)", nopython=True, nogil=True)
def hillshade(model, zenith, ray, res, ystart, yend):
    """returns a shaded region for model[ystart:yend]"""
    shadow = np.zeros((yend-ystart, model.shape[1]), dtype=np.int64)
    zenith = np.deg2rad(90 - zenith)
    dx, dy = ray
    xres, yres = res
    z_max = model.max()
    bounds = model.shape

    for y0 in range(ystart, yend):
        for x0 in range(model.shape[1]):
            z0 = model[y0, x0]
            x = float(x0)
            y = float(y0)

            intersection = None
            while within_bounds(x, y, bounds):
                z = elevation((x-x0)*xres, (y-y0)*yres, z0, zenith)
                if z > z_max:
                    break
                if z < model[int(y), int(x)]:
                    intersection = (y, x)
                    break
                x += dx;
                y += dy

            if intersection is not None:
                shadow[y0-ystart, x0] = 1
    return shadow
