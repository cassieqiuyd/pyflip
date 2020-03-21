from grid import Grid3D
import numpy as np
from utils import *

np.seterr(divide='ignore', invalid='ignore')

class Particles(object):
    def __init__(self, grid, simType):
        self.grid = grid
        self.particles = []
        self.n = 0
        self.x = []
        self.u = []
        self.cx = []
        self.cy = []
        self.cz = []
        self.simType = simType
        self.n_sum = np.zeros((self.grid.nx+1, self.grid.ny+1, self.grid.nz+1))

    def add_particle(self, px, pu):
        """
        :param px: position (x,y,z) of each particle
        :param pu: velocity (u,v,w) of each particle
        :return: None
        """
        self.x.append(px)
        self.u.append(pu)
        self.cx.append(np.zeros(3))
        self.cy.append(np.zeros(3))
        self.cz.append(np.zeros(3))
        self.n += 1

    def accumulate(self, accum, q, i, j, k, fx, fy, fz):
        diffx = 1-fx
        diffy = 1-fy
        diffz = 1-fz

        weight = diffx * diffy * diffz
        accum[i,j,k] += weight * q
        self.n_sum[i,j,k] += weight

        weight = fx * diffy * diffz
        accum[i+1,j,k] += weight * q
        self.n_sum[i+1,j,k] += weight

        weight = diffx * fy * diffz
        accum[i,j+1,k] += weight * q
        self.n_sum[i,j+1,k] += weight

        weight = fx * fy * diffz
        accum[i+1,j+1,k] += weight * q
        self.n_sum[i+1,j+1,k] += weight

        weight = diffx * diffy * fz
        accum[i,j,k+1] += weight * q
        self.n_sum[i,j,k+1] += weight

        weight = fx * diffy * fz
        accum[i+1,j,k+1] += weight * q
        
        self.n_sum[i+1,j,k+1] += weight

        weight = diffx * fy * fz
        accum[i,j+1,k+1] += weight * q
        self.n_sum[i,j+1,k+1] += weight

        weight = fx * fy * fz
        accum[i+1, j+1, k+1] += weight * q
        self.n_sum[i+1, j+1, k+1] += weight

        return accum

    def affineFix(self, accum, c, i, j, k, fx, fy, fz):
        diffx = 1 - fx
        diffy = 1 - fy
        diffz = 1 - fz

        weight = diffx * diffy * diffz
        accum[i,j,k] += weight * np.dot(c, (np.array([-fx,-fy,-fz]) * self.grid.h))

        weight = fx * diffy * diffz
        accum[i+1,j,k] += weight * np.dot(c, (np.array([diffx, -fy, -fz]) * self.grid.h))

        weight = diffx * fy * diffz
        accum[i,j+1,k] += weight * np.dot(c, (np.array([-fx, diffy, -fz]) * self.grid.h))

        weight = fx * fy * diffz
        accum[i+1, j+1, k] += weight * np.dot(c, (np.array([diffx, diffy, -fz]) * self.grid.h))

        weight = diffx * diffy * fz
        accum[i, j, k+1] += weight * np.dot(c, (np.array([-fx, -fy, diffz]) * self.grid.h))

        weight = fx * diffy * fz
        accum[i+1, j, k+1] += weight * np.dot(c, (np.array([diffx, -fy, diffz]) * self.grid.h))

        weight = diffx * fy * fz
        accum[i, j+1, k+1] += weight * np.dot(c, (np.array([-fx, diffy, diffz]) * self.grid.h))

        weight = fx * fy * fz
        accum[i+1, j+1, k+1] += weight * np.dot(c, (np.array([diffx, diffy, diffz]) * self.grid.h))

        return accum

    def transfer_to_grid(self):
        self.grid.u = np.zeros((self.grid.nx+1, self.grid.ny, self.grid.nz))
        self.n_sum = np.zeros(self.grid.u.shape)
        for p in range(self.n):
            ui, ufx = bary_x(self.x[p][0], self.grid.h)
            j, fy = bary_y_center(self.x[p][1], self.grid.h, self.grid.ny)
            k, fz = bary_z_center(self.x[p][2], self.grid.h, self.grid.nz)
            self.grid.u = self.accumulate(self.grid.u, self.u[p][0], ui, j, k, ufx, fy, fz)
            if (self.simType == "APIC"):
                self.grid.u = self.affineFix(self.grid.u, self.cx[p], ui, j, k, ufx, fy, fz)
        self.grid.u[np.where(self.n_sum != 0)] /= self.n_sum[np.where(self.n_sum != 0)]

        self.grid.v = np.zeros((self.grid.nx, self.grid.ny+1, self.grid.nz))
        self.n_sum = np.zeros(self.grid.v.shape)
        for p in range(self.n):
            i, fx = bary_x_center(self.x[p][0], self.grid.h, self.grid.nx)
            vj, vfy = bary_y(self.x[p][1], self.grid.h)
            k, fz = bary_z_center(self.x[p][2], self.grid.h, self.grid.nz)
            self.grid.v = self.accumulate(self.grid.v, self.u[p][1], i, vj, k, fx, vfy, fz)
            if (self.simType == "APIC"):
                self.grid.v = self.affineFix(self.grid.v, self.cy[p], i, vj, k, fx, vfy, fz)
        self.grid.v[np.where(self.n_sum != 0)] /= self.n_sum[np.where(self.n_sum != 0)]
        # for j in range(self.grid.ny):
        #     for i in range(self.grid.nx):
        #         for k in range(self.grid.nz):
        #             if self.n_sum[i,j,k] != 0:
        #                 self.grid.v[i,j,k] /= self.n_sum[i,j,k]

        self.grid.w = np.zeros((self.grid.nx, self.grid.ny, self.grid.nz+1))
        self.n_sum = np.zeros(self.grid.w.shape)
        for p in range(self.n):
            i, fx = bary_x_center(self.x[p][0], self.grid.h, self.grid.nx)
            j, fy = bary_y_center(self.x[p][1], self.grid.h, self.grid.ny)
            wk, wfz = bary_z(self.x[p][2], self.grid.h)
            self.grid.w = self.accumulate(self.grid.w, self.u[p][2], i, j, wk, fx, fy, wfz)
            if (self.simType == "APIC"):
                self.grid.w = self.affineFix(self.grid.w, self.cz[p], i, j, wk, fx, fy, wfz)
        self.grid.w[np.where(self.n_sum != 0)] /= self.n_sum[np.where(self.n_sum != 0)]

        # for j in range(self.grid.ny):
        #     for i in range(self.grid.nx):
        #         for k in range(self.grid.nz):
        #             if self.n_sum[i,j,k] != 0:
        #                 self.grid.w[i,j,k] /= self.n_sum[i,j,k]

        self.grid.marker = np.zeros(self.grid.size, dtype = int) #dim
        for p in range(self.n):
            i, fx = bary_x(self.x[p][0], self.grid.h)
            j, fy = bary_y(self.x[p][1], self.grid.h)
            k, fz = bary_z(self.x[p][2], self.grid.h)
            self.grid.marker[i, j, k] = 1

    def computeC(self, ufield, i, j, k, fx, fy, fz):
        diff_z = 1 - fz
        diff_y = 1 - fy
        diff_x = 1 - fx

        newC = np.zeros(3)

        #i, j, k
        weight = diff_x * diff_y * diff_z
        weight_prime = np.array([-fx, -fy, -fz]) * self.grid.h
        newC += weight * weight_prime * ufield[i, j, k]

        #i + 1, j, k
        weight = fx * diff_y * diff_z
        weight_prime = np.array([diff_x, -fy, -fz]) * self.grid.h
        newC += weight * weight_prime * ufield[i + 1, j, k]

        #i, j + 1, k
        weight = diff_x * fy * diff_z
        weight_prime = np.array([-fx, diff_y, -fz]) * self.grid.h
        newC += weight * weight_prime * ufield[i, j + 1, k]

        #i + 1, j + 1, k
        weight = fx * fy * diff_z
        weight_prime = np.array([diff_x, diff_y, -fz]) * self.grid.h
        newC += weight * weight_prime * ufield[i + 1, j + 1, k]

        # i, j, k + 1
        weight = diff_x * diff_y * fz
        weight_prime = np.array([-fx, -fy, diff_z]) * self.grid.h
        newC += weight * weight_prime * ufield[i, j, k + 1]

        # i + 1, j, k
        weight = fx * diff_y * fz
        weight_prime = np.array([diff_x, -fy, diff_z]) * self.grid.h
        newC += weight * weight_prime * ufield[i + 1, j, k + 1]

        # i, j + 1, k
        weight = diff_x * fy * fz
        weight_prime = np.array([-fx, diff_y, diff_z]) * self.grid.h
        newC += weight * weight_prime * ufield[i, j + 1, k + 1]

        # i + 1, j + 1, k
        weight = fx * fy * fz
        weight_prime = np.array([diff_x, diff_y, diff_z]) * self.grid.h
        newC += weight * weight_prime * ufield[i + 1, j + 1, k + 1]

        return newC

    def update_from_grid(self):
        for p in range(self.n):
            ui, ufx = bary_x(self.x[p][0], self.grid.h)
            i, fx = bary_x_center(self.x[p][0], self.grid.h, self.grid.nx)
            vj, vfy = bary_y(self.x[p][1], self.grid.h)
            j, fy = bary_y_center(self.x[p][1], self.grid.h, self.grid.ny)
            wk, wfz = bary_z(self.x[p][2], self.grid.h)
            k, fz = bary_z_center(self.x[p][2], self.grid.h, self.grid.nz)

            if self.simType == "FLIP":
                self. u[p] += np.array([trilerp(self.grid.du, ui, j, k, ufx, fy, fz),
                                        trilerp(self.grid.dv, i, vj, k, fx, vfy, fz),
                                        trilerp(self.grid.dw, i, j, wk, fx, fy, wfz)]) #FLIP
            else:
                self.u[p]= np.array([trilerp(self.grid.u, ui, j, k, ufx, fy, fz),
                                     trilerp(self.grid.v, i, vj, k, fx, vfy, fz),
                                     trilerp(self.grid.w, i, j, wk, fx, fy, wfz)]) #PIC and APIC

            if self.simType == "APIC":
                self.cx[p] = self.computeC(self.grid.u, ui, j, k, ufx, fy, fz)
                self.cy[p] = self.computeC(self.grid.v, i, vj, k, fx, vfy, fz)
                self.cz[p] = self.computeC(self.grid.w, i, j, wk, fx, fy, wfz)

    def move_particles_in_grid(self, dt):
        gu = np.zeros(3)
        x_min, x_max = 1.001 * self.grid.h, self.grid.lx - 1.001 * self.grid.h
        y_min, y_max = 1.001 * self.grid.h, self.grid.ly - 1.001 * self.grid.h
        z_min, z_max = 1.001 * self.grid.h, self.grid.lz - 1.001 * self.grid.h
        for p in range(self.n): # Runge-Kutta 2
            gu[0], gu[1], gu[2] = self.grid.trilerp_uvw(self.x[p][0], self.x[p][1], self.x[p][2])
            midx = self.x[p] + 0.5 * dt * gu
            np.clip(midx[0], x_min, x_max)
            np.clip(midx[1], y_min, y_max)
            np.clip(midx[2], z_min, z_max)
            #second stage of Runge - Kutta2

            if np.isnan(midx[0]):
                print(self.x[p], midx, gu)
            gu[0], gu[1], gu[2] = self.grid.trilerp_uvw(midx[0], midx[1], self.x[p][2]) #should it be this or midx[2]
            self.x[p] += dt * gu
            np.clip(self.x[p][0], x_min, x_max)
            np.clip(self.x[p][1], y_min, y_max)
            np.clip(self.x[p][2], z_min, z_max)

    def write_to_file(self, filename):
        '''
            create a new file where each particle's position is
            stored. This method will first create a new file specify
            how many particles there are and then, for each particle,
            specify the x y z coordinates of the particle
            Input:
                filename: name of file where the particle coordinates are written to
            Output:
                None. Will have a file created with the particle coordinates
        '''
        f = open(filename, "w+")
        f.write(str(self.n) + '\n')
        for p in range(0, self.n):
            particle_coor = str(self.x[p][0]) + " " + str(self.x[p][1]) + " " + str(self.x[p][2]) + "\n"
            f.write(particle_coor)
        f.close()