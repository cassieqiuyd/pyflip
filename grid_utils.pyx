import numpy as np
# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport numpy as np

DTYPE = np.float
# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.float_t DTYPE_t

# ======= Sweep Functions ======== #

cdef np.float_t distance(np.float_t p, np.float_t q, np.float_t t):
    cdef np.float_t pq_min = min(p,q)
    cdef np.float_t pq_max = max(p,q);
    cdef np.float_t min3 = min(t,pq_min)
    cdef np.float_t max3 = max(t,pq_max)
    cdef np.float_t mid3 = max(pq_min,max(pq_max,t))

    cdef np.float_t d = min3 + 1
    if d > mid3:
        d = (min3+mid3+np.sqrt(2-np.sqrt(min3-mid3)))/2
        if d > max3: 
            d = (min3+mid3+max3+np.sqrt(3-2*(min3*min3-min3*mid3-min3*max3+mid3*mid3-mid3*max3+max3*max3)))/3
    return d

def fast_sweep_methods_phi(np.ndarray[np.float_t, ndim=3] phi, np.ndarray[np.int_t, ndim=3] marker):
    assert phi.shape[0] == marker.shape[0]
    assert phi.shape[1] == marker.shape[1]
    assert phi.shape[2] == marker.shape[2]
    
    cdef int umax = phi.shape[0]
    cdef int vmax = phi.shape[1]
    cdef int wmax = phi.shape[2]
    cdef np.float_t d

    for k in range(1, wmax):
        for j in range(1, vmax):
            for i in range(1, umax):
                if marker[i, j, k] == 0:
                    d = distance(phi[i-1, j, k], phi[i, j-1, k], phi[i, j, k-1])
                    phi[i, j, k] = min(d, phi[i, j, k])
    for k in reversed(range(1, wmax)):
        for j in range(1, vmax):
            for i in range(1, umax):
                if marker[i, j, k] == 0:
                    d = distance(phi[i-1, j, k], phi[i, j-1, k], phi[i, j, k-1])
                    phi[i, j, k] = min(d, phi[i, j, k])
    for k in range(1, wmax):
        for j in reversed(range(1, vmax-1)):
            for i in range(1, umax):
                if marker[i, j, k] == 0:
                    d = distance(phi[i-1, j, k], phi[i, j-1, k], phi[i, j, k-1])
                    phi[i, j, k] = min(d, phi[i, j, k])
    for k in range(1, wmax):
        for j in range(1, vmax):
            for i in reversed(range(1, umax)):
                if marker[i, j, k] == 0:
                    d = distance(phi[i-1, j, k], phi[i, j-1, k], phi[i, j, k-1])
                    phi[i, j, k] = min(d, phi[i, j, k])
    for k in reversed(range(1, wmax)):
        for j in reversed(range(1, vmax)):
            for i in range(1, umax):
                if marker[i, j, k] == 0:
                    d = distance(phi[i-1, j, k], phi[i, j-1, k], phi[i, j, k-1])
                    phi[i, j, k] = min(d, phi[i, j, k])
    for k in reversed(range(1, wmax)):
        for j in range(1, vmax):
            for i in reversed(range(1, umax)):
                if marker[i, j, k] == 0:
                    d = distance(phi[i-1, j, k], phi[i, j-1, k], phi[i, j, k-1])
                    phi[i, j, k] = min(d, phi[i, j, k])
    for k in range(1, wmax):
        for j in reversed(range(1, vmax)):
            for i in reversed(range(1, umax)):
                if marker[i, j, k] == 0:
                    d = distance(phi[i-1, j, k], phi[i, j-1, k], phi[i, j, k-1])
                    phi[i, j, k] = min(d, phi[i, j, k])
    for k in reversed(range(1, wmax)):
        for j in reversed(range(1, vmax)):
            for i in reversed(range(1, umax)):
                if marker[i, j, k] == 0:
                    d = distance(phi[i-1, j, k], phi[i, j-1, k], phi[i, j, k-1])
                    phi[i, j, k] = min(d, phi[i, j, k])
    return phi

def sweep_u(int i0, int i1, int j0, int j1, int k0, int k1, np.ndarray u, np.ndarray phi, np.ndarray marker):
    cdef int di = np.sign(i1-i0) * 1
    cdef int dj = np.sign(j1-j0) * 1
    cdef int dk = np.sign(k1-k0) * 1
    cdef float dp, dq, dr, alpha, beta
    for k in np.arange(k0, k1)[::dk]:
        for j in np.arange(j0, j1)[::dj]:
            for i in np.arange(i0, i1)[::di]:
                if marker[i-1,j,k]==0 and marker[i,j,k]==0:
                    dq = dj * (phi[i,j,k]-phi[i,j-1,k])
                    if dq < 0: continue
                    dp = 0.5 * (phi[i,j-1,k]+phi[i,j,k]-phi[i-di,j-1,k]-phi[i-di,j,k])
                    if dp < 0: continue
                    dr = 0.5 * (phi[i-1,j,k]+phi[i,j,k]-phi[i-1,j,k-dk]-phi[i,j,k-dk])
                    if dr < 0: continue
                    if dp+dq+dr == 0: 
                        alpha = 1/3.0
                    else:
                        alpha = dp/(dp+dq+dr)
                        beta = dq/(dp+dq+dr)
                    u[i,j,k] = alpha * u[i-di,j,k] + beta * u[i,j-dj,k] + (1-alpha-beta) * u[i,j,k-dk]
    return u

def sweep_v(int i0, int i1, int j0, int j1, int k0, int k1, np.ndarray v, np.ndarray phi, np.ndarray marker):
    cdef int di = np.sign(i1-i0) * 1
    cdef int dj = np.sign(j1-j0) * 1
    cdef int dk = np.sign(k1-k0) * 1
    cdef float dp, dq, dr, alpha, beta
    for k in np.arange(k0, k1)[::dk]:
        for j in np.arange(j0, j1)[::dj]: 
            for i in np.arange(i0, i1)[::di]:
                if marker[i,j-1,k]==0 and marker[i,j,k]==0: 
                    dq = dj * (phi[i,j,k]-phi[i,j-1,k])
                    if dq < 0: continue
                    dp = 0.5 * (phi[i,j-1,k]+phi[i,j,k]-phi[i-di,j-1,k]-phi[i-di,j,k])
                    if dp < 0: continue
                    dr = 0.5 * (phi[i-1,j,k]+phi[i,j,k]-phi[i-1,j,k-dk]-phi[i,j,k-dk])
                    if dr < 0: continue
                    if dp+dq+dr == 0: 
                        alpha=1/3.0
                    else:
                        alpha=dp/(dp+dq+dr)
                        beta=dq/(dp+dq+dr)
                    v[i,j,k] = alpha * v[i-di,j,k] + beta * v[i,j-dj,k] + (1-alpha-beta) * v[i,j,k-dk]
    return v

def sweep_w(int i0, int i1, int j0, int j1, int k0, int k1, np.ndarray w, np.ndarray phi, np.ndarray marker):
    cdef int di = np.sign(i1-i0) * 1
    cdef int dj = np.sign(j1-j0) * 1
    cdef int dk = np.sign(k1-k0) * 1
    cdef float dp, dq, dr, alpha, beta
    for k in np.arange(k0, k1)[::dk]:
        for j in np.arange(j0, j1)[::dj]: 
            for i in np.arange(i0, i1)[::di]:
                if marker[i,j,k-1]==0 and marker[i,j,k]==0: 
                    dq = dj * (phi[i,j,k]-phi[i,j,k-1])
                    if dq < 0: continue
                    dp = 0.5 * (phi[i,j-1,k]+phi[i,j,k]-phi[i,j-1,k-dk]-phi[i,j-1,k-dk])
                    if dp < 0: continue
                    dr = 0.5 * (phi[i-1,j,k]+phi[i,j,k]-phi[i-1,j,k-dk]-phi[i-1,j,k-dk])
                    if dr < 0: continue
                    if dp+dq+dr == 0: 
                        alpha=1/3.0
                    else:
                        alpha=dp/(dp+dq+dr)
                        beta=dq/(dp+dq+dr)
                    w[i,j,k] = alpha * w[i-di,j,k] + beta * w[i,j-dj,k] + (1-alpha-beta) * w[i,j,k-dk]
    return w