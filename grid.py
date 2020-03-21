import copy
from grid_utils import *
import numpy as np
from utils import *

def sweep(x, y, z, x_desc=False, y_desc=False, z_desc=False):
    inds = sorted(sorted(sorted([(i, j, k) for i, j, k in zip(x,y,z)], key=lambda a: a[0], reverse=x_desc), key=lambda a: a[1], reverse=y_desc), key=lambda a: a[2], reverse=z_desc)
    return inds

class Grid3D:
    """
    Variables:
    - gravity
    - lx, ly
    - h, overh

    Tracking State Variables:
    - u, v  # staggered MAC facet fluid velocities
    - du, dv # saved velocities for particle update
    - marker
        types of material represented in the marker:
        - FLUID: 1
        - EMPTY (AIR): 0
        - SOLID: -1
    - phi # decays away from water into air (for extrapolating velocity)
    - pressure

    Poisson Equation Solver:
    - poisson -- for the neighborhood connectivity matrix
              -- [N x 3], 0 = number of non-solid neighbor, 1 = right neighbor is fluid, 2 = top neighbor is fluid
    - preconditioner -- L for the cholesky decomposition
    General Form of Equation:
    [ Poisson ] [ Pressure ] = [ Divergence ]
    """
    def __init__(self, gravity, size, lx):
        nx, ny, nz= size
        self.nx = nx
        self.ny = ny
        self.nz = nz
        """
        Basic Layout of the MAC grid:
        [u_11, v_11] || [u_12, v_12] || [u_13, v_13]
            ||     p_11 	   ||     p_12     ||
        [u_21, v_21] || [u_22, v_22] || [u_23, v_23]
        """
        self.size = size
        self.u = np.zeros((nx+1, ny, nz))
        self.v = np.zeros((nx, ny+1, nz))
        self.w = np.zeros((nz, ny, nz+1))
        self.du = np.zeros((nx+1, ny, nz))
        self.dv = np.zeros((nx, ny+1, nz))
        self.dw = np.zeros((nx, ny, nz+1))

        # center of the marker cube
        self.pressure = np.zeros(size)
        self.marker = np.zeros(size, dtype=int)

        # boundary condition (solid)
        self.marker[0, :, :] = self.marker[-1, :, :] = -1
        self.marker[:, 0, :] = self.marker[:, -1, :] = -1
        self.marker[:, :, 0] = self.marker[:, :, -1] = -1
        
        self.u[0, :, :] = self.u[1, :, :] = self.u[-1, :, :] = self.u[-2, :, :] = 0
        self.v[:, 0, :] = self.v[:, 1, :] = self.v[:, -1, :] = self.v[:, -2, :] = 0
        self.w[:, :, 0] = self.w[:, :, 1] = self.w[:, :, -1] = self.w[:, :, -2] = 0

        # physical simulation variable for solving poisson equation
        self.gravity = gravity
        self.lx, self.ly, self.lz = lx, ny / nx * lx, nz / nx * lx
        # individual MAC cell height
        self.h = lx / nx

        # phi is distance to fluid
        self.phi = np.zeros(size)
        self.poisson = np.zeros(size)
        self.preconditioner = np.zeros(size)
        # intermediate matrix - m = cholesky Lx, r = distance, div = divergence, div_p = divergence_prime
        self.m, self.r, self.div, self.div_p = np.zeros(size), np.zeros(size), np.zeros(size), np.zeros(size)

    def apply_boundary_conditions(self):
        self.marker[0, :, :] = self.marker[-1, :, :] = -1
        self.marker[:, 0, :] = self.marker[:, -1, :] = -1
        self.marker[:, :, 0] = self.marker[:, :, -1] = -1

        self.u[0, :, :] = self.u[1, :, :] = self.u[-1, :, :] = self.u[-2, :, :] = 0
        self.v[:, 0, :] = self.v[:, 1, :] = self.v[:, -1, :] = self.v[:, -2, :] = 0
        self.w[:, :, 0] = self.w[:, :, 1] = self.w[:, :, -1] = self.w[:, :, -2] = 0

    def CFL(self):
        maxv3 = max(self.h * self.gravity, infnorm(self.u) **2 + infnorm(self.v) **2 + infnorm(self.w) **2)
        if maxv3 < 1e-16:
            maxv3 = 1e-16
        return self.h / np.sqrt(maxv3)

    def make_incompressible(self):
        """
        Adjusting the fluid velocities to make the system conforms with Navier Stoke Incompressible Fluid Assumption
        """
        self.find_divergence()
        poisson = self.get_poisson()
        preconditioner = self.get_preconditioner(poisson)
        self.solve_pressure(100, 1e-5, poisson, preconditioner)
        self.add_gradient()

    def save_velocities(self):
        self.u = copy.deepcopy(self.du)
        self.v = copy.deepcopy(self.dv)
        self.w = copy.deepcopy(self.dw)

    def get_velocity_update(self):
        self.du = self.u - self.du
        self.dv = self.v - self.dv
        self.dw = self.w - self.dw

    def add_gravity(self, dt):
        # uniformly add gravity
        dtg = dt * self.gravity
        self.v -= dtg

    def compute_distance_to_fluid(self):
        self.init_phi()
        self.sweep_phi()

    def extend_velocity(self):
        for i in range(8):
            self.sweep_velocity()

    # ===== Navier Stoke Equation for Solving Pressure and Velocities Update ===== #

    def find_divergence(self):
        """
        finding the difference between the two directional velocities
        """
        self.div = np.zeros(self.marker.shape)
        dx, dy, dz = np.where(self.marker == 1)
        # calculate divergence at three different directions
        self.div[dx, dy, dz] = self.u[dx+1, dy, dz] - self.u[dx, dy, dz] \
                               + self.v[dx, dy + 1, dz] - self.v[dx, dy, dz]\
                               + self.w[dx, dy, dz+1] - self.w[dx, dy, dz]

    def get_poisson(self):
        """
        poisson matrix encodes 3-d vector:
            [x, y, z] - x, the number of non-solid neighbor cells, y, is right cell a fluid cell z, is top cell a fluid cell
        """
        poisson = np.zeros((self.marker.shape[0], self.marker.shape[1], self.marker.shape[2], 4))
        fx, fy, fz = np.where(self.marker == 1)
        poisson[fx, fy, fz, 0] = (self.marker[fx-1, fy, fz] != -1) + (self.marker[fx+1, fy, fz] != -1) \
                                 + (self.marker[fx, fy-1, fz] != -1) + (self.marker[fx, fy+1, fz] != -1)\
                                 + (self.marker[fx, fy, fz-1] != -1) + (self.marker[fx, fy, fz-1] != -1)
        poisson[fx, fy, fz, 1] = (self.marker[fx+1, fy, fz] == 1)
        poisson[fx, fy, fz, 2] = (self.marker[fx, fy+1, fz] == 1)
        poisson[fx, fy, fz, 3] = (self.marker[fx, fy, fz+1] == 1)
        return poisson


    def get_preconditioner(self, poisson):
        """
        Find L by performing the cholesky decomposition of poisson matrix:
            - same shape as poisson matrix
        """
        preconditioner = np.zeros(self.marker.shape)
        lam = 0.99
        fx, fy, fz = np.where(self.marker == 1)
        for i, j, k in sweep(fx, fy, fz, x_desc=False, y_desc=False, z_desc=False):
            d = poisson[i, j, k, 0] \
                - np.sqrt(poisson[i-1, j, k, 1] * preconditioner[i-1, j, k]) \
                - np.sqrt(poisson[i, j-1, k, 2] * preconditioner[i, j-1, k]) \
                - np.sqrt(poisson[i, j, k-1, 3] * preconditioner[i, j, k-1]) \
                - lam * (poisson[i-1, j, k, 1] * (poisson[i-1, j, k, 2] + poisson[i-1, j, k, 3]) * np.sqrt(preconditioner[i-1, j, k])
                         + poisson[i, j-1, k, 2] * (poisson[i, j-1, k, 1] + poisson[i, j-1, k, 3]) * np.sqrt(preconditioner[i, j-1, k])
                         + poisson[i, j, k-1, 3] * (poisson[i, j, k-1, 1] + poisson[i, j, k-1, 2]) * np.sqrt(preconditioner[i, j, k-1]))
            preconditioner[i, j, k] = 1/np.sqrt(d + 1e-6)
        return preconditioner

    def apply_poisson(self, x, poisson):
        """
        :param: x, the input of the poisson matrix,
        :return: y, the matrix multiplication output of the poisson matrix
        """
        y = np.zeros(self.marker.shape)
        fx, fy, fz = np.where(self.marker == 1)
        y[fx, fy, fz] = poisson[fx, fy, fz, 0] * x[fx, fy, fz]\
                        + poisson[fx-1, fy, fz, 1] * x[fx-1, fy, fz] \
                        + poisson[fx, fy-1, fz, 2] * x[fx, fy-1, fz] \
                        + poisson[fx, fy, fz-1, 3] * x[fx, fy, fz-1] \
                        + poisson[fx, fy, fz, 1] * x[fx+1, fy, fz] \
                        + poisson[fx, fy, fz, 2] * x[fx, fy+1, fz] \
                        + poisson[fx, fy, fz, 3] * x[fx, fy, fz+1]
        return y

    # Cython
    def apply_preconditioner(self, x, poisson, preconditioner):
        """
        Gauss-elimination to solve triangle matrix inversion
        """
        m = np.zeros(self.marker.shape)
        # solve L m = x, to get m
        fx, fy, fz = np.where(self.marker == 1)
        for i, j, k in sweep(fx, fy, fz, x_desc=False, y_desc=False, z_desc=False):
            d = x[i, j, k] - poisson[i-1, j, k, 1] * preconditioner[i-1, j, k] * m[i-1, j, k] \
                - poisson[i, j-1, k, 2] * preconditioner[i, j-1, k] * m[i, j-1, k] \
                - poisson[i, j, k-1, 3] * preconditioner[i, j, k-1] * m[i, j, k-1]
            m[i, j, k] = preconditioner[i, j, k] * d
        y = np.zeros(self.marker.shape)
        for i, j, k in sweep(fx, fy, fz, x_desc=True, y_desc=True, z_desc=True):
            d = m[i, j, k] - poisson[i, j, k, 1] * preconditioner[i, j, k] * y[i+1, j, k] \
                           - poisson[i, j, k, 2] * preconditioner[i, j, k] * y[i, j+1, k] \
                           - poisson[i, j, k, 3] * preconditioner[i, j, k] * y[i, j, k+1]
            y[i, j, k] = preconditioner[i, j, k] * d
        # L L' y = x
        return y

    def solve_pressure(self, max_iter, tol, poisson, preconditioner):
        """
        Iterative method for solving pressure and divergence:
        - y: variable keeps track of pressure, ys: variable keeps track of divergence
        """
        # calculate the infinite norm of divergence for all the cells
        tol_m = tol * np.max(np.absolute(self.div))
        pressure = np.zeros(self.marker.shape)
        if np.max(np.absolute(self.div)) == 0:
            return
        # solve pressure - pressure = poisson^(-1) * divergence
        y = self.apply_preconditioner(self.div, poisson, preconditioner)
        ys = copy.deepcopy(y)
        # y (1 x n) and div (n x 1)
        rho = np.sum(y * self.div)
        if rho == 0:
            return
        for i in range(0, max_iter):
            # solve divergence - new divergence = poisson * pressure
            y = self.apply_poisson(ys, poisson)
            # error e^Te
            alpha = rho / (ys @ y)
            pressure += alpha * ys
            self.div += -alpha * y
            if np.max(np.absolute(self.div)) <= tol:
                print("pressure converged to {} in {} iterations".format(np.max(np.absolute(self.div)), i))
                return
            y = self.apply_preconditioner(self.div, poisson, preconditioner)
            rhonew = np.sum(y * self.div)
            beta = rhonew / rho
            ys = beta * ys + y
            rho = rhonew
        print("Didn't converge in pressure solve (its={}, tol={}, |r|={})\n".format(i,tol_m,np.max(abs(self.div))))

    def add_gradient(self):
        fx, fy, fz = np.where(self.marker == 1)
        # neighbor cannot be solid
        mask = (self.marker[fx-1, fy, fz] != -1)
        self.u[fx, fy, fz] += (self.pressure[fx, fy, fz] - self.pressure[fx-1, fy, fz]) * mask
        # neighbor cannot be solid
        mask = (self.marker[fx, fy-1, fz] != -1)
        self.v[fx, fy, fz] += (self.pressure[fx, fy, fz] - self.pressure[fx, fy-1, fz]) * mask
        # neighbor cannot be solid
        mask = (self.marker[fx, fy, fz-1] != -1)
        self.w[fx, fy, fz] += (self.pressure[fx, fy, fz] - self.pressure[fx, fy, fz-1]) * mask

    # ===== Compute AIR Cell Velocity Update ===== #

    def init_phi(self):
        # large distance: nx + ny + 2
        self.phi[:, :, :] = (self.phi.shape[0] + self.phi.shape[1]+self.phi.shape[2] + 3)
        # start off with indicator inside the fluid and overestimates of distance outside
        fx, fy, fz = np.where(self.marker == 1)
        self.phi[fx, fy, fz] = -0.5

    def sweep_phi(self):
        """
        Compute the distance from air to fluid cell using FAST SWEEPING ALGORITHM
        Eikonal Equation for Numerical Approximation:
        - https://en.wikipedia.org/wiki/Eikonal_equation#Numerical_approximation
        """
        self.phi = fast_sweep_methods_phi(self.phi, self.marker)

    def sweep_velocity(self):
        nx, ny, nz = self.nx, self.ny, self.nz
        self.u = sweep_u(1, nx-1, 1, ny-1, 1, nz-1, self.u, self.phi, self.marker)
        self.u = sweep_u(1, nx-1, ny-2, 0, 1, nz-1, self.u, self.phi, self.marker)
        self.u = sweep_u(nx-2, 0, 1, ny-1, 1, nz-1, self.u, self.phi, self.marker)
        self.u = sweep_u(nx-2, 0, ny-2, 0, 1, nz-1, self.u, self.phi, self.marker)
        self.u = sweep_u(1, nx-1, 1, ny-1, nz-2, 0, self.u, self.phi, self.marker)
        self.u = sweep_u(1, nx-1, ny-2, 0, nz-2, 0, self.u, self.phi, self.marker)
        self.u = sweep_u(nx-2, 0, 1, ny-1, nz-2, 0, self.u, self.phi, self.marker)
        self.u = sweep_u(nx-2, 0, ny-2, 0, nz-2, 0, self.u, self.phi, self.marker)
        self.u[:, 0, :] = self.u[:, 1, :]
        self.u[:, ny-1, :] = self.u[:, ny-2, :]
        self.u[0, :, :] = self.u[1, :, :]
        self.u[nx-1, :, :] = self.u[nx-2, :, :]
        self.u[:, :, 0] = self.u[:, :, 1]
        self.u[:, :, nz-1] = self.u[:, :, nz-2]
        self.v = sweep_v(1, nx-1, 1, ny-1, 1, nz-1, self.v, self.phi, self.marker)
        self.v = sweep_v(1, nx-1, ny-2, 0, 1, nz-1, self.v, self.phi, self.marker)
        self.v = sweep_v(nx-2, 0, 1, ny-1, 1, nz-1, self.v, self.phi, self.marker)
        self.v = sweep_v(nx-2, 0, ny-2, 0, 1, nz-1, self.v, self.phi, self.marker)
        self.v = sweep_v(1, nx-1, 1, ny-1, nz-2, 0, self.v, self.phi, self.marker)
        self.v = sweep_v(1, nx-1, ny-2, 0, nz-2, 0, self.v, self.phi, self.marker)
        self.v = sweep_v(nx-2, 0, 1, ny-1, nz-2, 0, self.v, self.phi, self.marker)
        self.v = sweep_v(nx-2, 0, ny-2, 0, nz-2, 0, self.v, self.phi, self.marker)
        self.v[:, 0, :] = self.v[:, 1, :]
        self.v[:, ny-1, :] = self.v[:, ny-2, :]
        self.v[0, :, :] = self.v[1, :, :]
        self.v[nx-1, :, :] = self.v[nx-2, :, :]
        self.v[:, :, 0] = self.v[:, :, 1]
        self.v[:, :, nz-1] = self.v[:, :, nz-2]
        self.w = sweep_w(1, nx-1, 1, ny-1, 1, nz-1, self.w, self.phi, self.marker)
        self.w = sweep_w(1, nx-1, ny-2, 0, 1, nz-1, self.w, self.phi, self.marker)
        self.w = sweep_w(nx-2, 0, 1, ny-1, 1, nz-1, self.w, self.phi, self.marker)
        self.w = sweep_w(nx-2, 0, ny-2, 0, 1, nz-1, self.w, self.phi, self.marker)
        self.w = sweep_w(1, nx-1, 1, ny-1, nz-2, 0, self.w, self.phi, self.marker)
        self.w = sweep_w(1, nx-1, ny-2, 0, nz-2, 0, self.w, self.phi, self.marker)
        self.w = sweep_w(nx-2, 0, 1, ny-1, nz-2, 0, self.w, self.phi, self.marker)
        self.w = sweep_w(nx-2, 0, ny-2, 0, nz-2, 0, self.w, self.phi, self.marker)
        self.w[:, 0, :] = self.w[:, 1, :]
        self.w[:, ny-1, :] = self.w[:, ny-2, :]
        self.w[0, :, :] = self.w[1, :, :]
        self.w[nx-1, :, :] = self.w[nx-2, :, :]
        self.w[:, :, 0] = self.w[:, :, 1]
        self.w[:, :, nz-1] = self.w[:, :, nz-2]

    # ===== trilinear interpolation ===== #
    def trilerp_uvw(self, px, py, pz):
        i, fx = bary_x(px, self.h)
        j, fy = bary_y_center(py, self.h, self.ny)
        k, fz = bary_z_center(pz, self.h, self.nz)

        pu = trilerp(self.u, i, j, k, fx, fy, fz)

        i, fx = bary_x_center(px, self.h, self.nx)
        j, fy = bary_y(py, self.h)
        k, fz = bary_z_center(pz, self.h, self.nz)

        pv = trilerp(self.v, i, j, k, fx, fy, fz)

        i, fx = bary_x_center(px, self.h, self.nx)
        j, fy = bary_y_center(py, self.h, self.ny)
        k, fz = bary_z(pz, self.h)

        pw = trilerp(self.w, i, j, k, fx, fy, fz)

        if np.isnan(pu):
            print('xyzis',px,py,pz)

        return pu, pv, pw
