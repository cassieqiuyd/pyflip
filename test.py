import numpy as np
from grid_utils import *
from grid import Grid3D

def main():
    grid =  Grid3D(9.8, (30, 30, 30), 1)
    a = grid.trilerp_uvw(0.51106383, 0.53480572, 0.50850903)
    print(a)

if __name__ == "__main__":
    main()