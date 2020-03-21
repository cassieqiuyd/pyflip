import numpy as np


def infnorm(a):
    return np.max(np.absolute(a))


def trilerp(u, i, j, k, fx, fy, fz):
    return (1-fz) * ((1-fy) * ((1-fx)*u[i, j, k] + fx*u[i+1, j, k])
                     + fy * ((1-fx)*u[i, j+1, k] + fx*u[i+1, j+1, k])) \
             + fz * ((1-fy) * ((1-fx)*u[i, j, k+1] + fx*u[i+1, j, k+1])
                     + fy * ((1-fx)*u[i, j+1, k+1] + fx*u[i+1, j+1, k+1]))


# ===== trilinear interpolation ===== #
def bary_x(x, h):
    overh = 1 / h
    sx = x * overh
    i = int(sx)
    fx = sx - np.floor(sx)
    return i, fx


def bary_x_center(x, h, nx):
    overh = 1 / h
    sx = x * overh
    i = int(sx)
    fx = sx - np.floor(sx)
    if i < 0:
        i, fx = 0, 0.0
    elif i > nx - 2:
        i = nx - 2
        fx = 1.0
    return i, fx


def bary_y(y, h):
    overh = 1 / h
    sy = y * overh
    j = int(sy)
    fy = sy - np.floor(sy)
    return j, fy


def bary_y_center(y, h, ny):
    overh = 1 / h
    sy = y * overh
    j = int(sy)
    fy = sy - np.floor(sy)
    if j < 0:
        j, fy = 0, 0.0
    elif j > ny - 2:
        j = ny - 2
        fy = 1.0
    return j, fy


def bary_z(z, h):
    overh = 1 / h
    sz = z * overh
    k = int(sz)
    fz = sz - np.floor(sz)
    return k, fz


def bary_z_center(z, h, nz):
    overh = 1 / h
    sz = z * overh
    k = int(sz)
    fz = sz - np.floor(sz)
    if k < 0:
        k, fz = 0, 0.0
    elif k > nz - 2:
        k = nz - 2
        fz = 1.0
    return k, fz
