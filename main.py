import numpy as np
import argparse
from grid import Grid3D
import os
from particles import Particles
from shared_main import *

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

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
def main():
    parser = argparse.ArgumentParser(description="FluidSim")
    parser.add_argument("--USE_SPHERICAL_GRAV", default=False)
    parser.add_argument("--SimType", type=str, default='PIC')
    parser.add_argument("--GRAV_FACTOR", type=float, default=0.01)
    parser.add_argument('--out_dir', type=str, default='output/')
    parser.add_argument('--n_iter', type=int, default=50)
    args = parser.parse_args()

    gravity = 9.8
    if (args.USE_SPHERICAL_GRAV):
        gravity *= args.GRAV_FACTOR

    grid = Grid3D(gravity, (30, 30, 30), 1)  #g, (nx,ny,nz),lx

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    particles = Particles(grid, args.SimType)
    init_water_drop(grid, particles, 2, 2, 2)
    particles.write_to_file("{}/frameparticles{:0>4}".format(args.out_dir, 0))

    for i in range(args.n_iter):
        print("===================================================> step {}...\n".format(i))
        advance_one_frame(grid, particles, 1/30)
        particles.write_to_file("{}/frameparticles{:0>4}".format(args.out_dir, i))


if __name__ == '__main__':
    main()