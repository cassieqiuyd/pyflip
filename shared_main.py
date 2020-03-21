import numpy as np
import random

SIMULATION_TYPE = "APIC"
INIT_DROP_RADIUS = 0.05
INIT_FLOOR_SIZE = 0.05
USE_SPHERICAL_GRAV = False

#the following only matter when USE_SPHERICAL_GRAV is true
INIT_VEL_MAGNITUDE = 0.55


def fluidphi(grid, x, y, z):
    """
    This sets the signed distance function phi of the fluid. You can selectively uncomment a line to decide
    used in the project function
    :return:
    """
    # return y-0.5*grid.ly  #no drop
    # return min(sqrt(sqr(x-0.5*grid.lx)+sqr(y-0.625*grid.ly))-0.02*grid.ly, y-0.6*grid.ly) #tiny drop
    # return min(sqrt(sqr(x-0.3333*grid.lx)+sqr(y-0.71*grid.ly))-0.3*grid.ly, y-0.2*grid.ly) #large drop
    # return max(y-0.8*grid.ly, -sqrt(sqr(x-0.5*grid.lx)+sqr(y-0.2*grid.ly))+0.1*grid.lx) #bubble
    # return sqrt(sqr(x-0.5*grid.lx)+sqr(y-0.75*grid.ly))-0.15*grid.lx #large drop without bottom
    # return 0.75*grid.lx-x  #vertical column
    # return max( max(x-0.75*grid.lx, 0.25*grid.lx-x), max(y-0.75*grid.ly, 0.25*grid.ly-y)) # small box
    return min(y - INIT_FLOOR_SIZE * grid.ly,
               np.sqrt((x - 0.5 * grid.lx) ** 2 + (y - 0.5 * grid.ly) ** 2 + (z - 0.5 * grid.lz) ** 2)
               - INIT_DROP_RADIUS * grid.lx)  # medium drop


def project(grid, x, y, z, current, target):
    """
    Helper function for initializing the water
    """
    dpdx = (fluidphi(grid, x + 1e-4, y, z) - fluidphi(grid, x - 1e-4, y, z)) / 2e-4
    dpdy = (fluidphi(grid, x, y + 1e-4, z) - fluidphi(grid, x, y - 1e-4, z)) / 2e-4
    dpdz = (fluidphi(grid, x, y, z + 1e-4) - fluidphi(grid, x, y, z - 1e-4)) / 2e-4
    scale = (target - current) / np.sqrt(dpdx * dpdx + dpdy * dpdy + dpdz * dpdz)
    x += scale * dpdx
    y += scale * dpdy
    z += scale * dpdz

    return x, y, z

def init_water_drop(grid, particles, na, nb, nc):
    """
    This function allocates particles based on the phi function
    :param grid:
    :param particles:
    :param na:
    :param nb:
    :param nc:
    :return:
    """
    vx = 0
    vy = 0
    vz = 0

    for i in range(grid.nx-1):
        for j in range(grid.ny-1):
            for k in range(grid.nz-1):
                for a in range(na):
                    for b in range(nb):
                        for c in range(nc):
                            x = (i + (a + 0.1 + 0.8 * random.random()) / na) * grid.h
                            y = (j + (b + 0.1 + 0.8 * random.random()) / nb) * grid.h
                            z = (k + (c + 0.1 + 0.8 * random.random()) / nc) * grid.h

                            phi = fluidphi(grid, x, y, z)
                            if phi > -0.25 * grid.h / na:
                                continue
                            elif phi > -1.5 * grid.h / na:
                                x, y, z = project(grid, x, y, z, phi, -0.75 * grid.h / na)
                                phi = fluidphi(grid, x, y, z)
                                x, y, z = project(grid, x, y, z, phi, -0.75 * grid.h / na)
                                phi = fluidphi(grid, x, y, z)
                                x, y, z = project(grid, x, y, z, phi, -0.75 * grid.h / na)
                                phi = fluidphi(grid, x, y, z)
                            particles.add_particle(np.array([x, y, z]), np.array([vx, vy, vz]))

"""
General Algorithm Outline:
	Initialize All the Particles
	Loop: 
		1) Particles => Move Particles
		2) Particles => Transfer Kinematics To MAC Grid
		3) MAC Grid  => Save Velocities
		4) MAC Grid  => Add External Forces (Gravity or Particle Collisions)
		5) MAC Grid  => Compute Distance To Fluid (FAST SWEEPING ALGORITHM)
		6) MAC Grid  => Extending Velocity Field (To AIR Cell)
		7) MAC Grid  => Apply Boundary Conditions
		8) MAC Grid  => Imposing Navier Stoke Incompressible Equation
		9) MAC Grid  => Extending Velocity Field (To AIR Cell)
		10) MAC Grid => Update the Velocity Field
		11) Particles=> Update Particle from MAC Grid
"""


def advance_one_step(grid, particles, dt):
    for i in range(5):
        particles.move_particles_in_grid(0.2 * dt)
    particles.transfer_to_grid()
    grid.save_velocities()
    grid.add_gravity(dt)
    grid.compute_distance_to_fluid()
    grid.extend_velocity()
    grid.apply_boundary_conditions()
    grid.make_incompressible()
    grid.extend_velocity()
    grid.get_velocity_update()
    particles.update_from_grid()


def advance_one_frame(grid, particles, frametime):
    t = 0
    finished = False

    while not finished:
        dt = 2 * grid.CFL()
        if t + dt > frametime:
            dt = frametime - t
            finished = True
        elif t+1.5*dt >= frametime:
            dt = 0.5 * (frametime - t)
        print("advancing {} (to {} of frame)\n".format(dt, 100.0 * (t + dt) / frametime))
        advance_one_step(grid, particles, dt)
        t += dt
        print(t)