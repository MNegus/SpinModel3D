'Test lattice models'

from __future__ import division

import numpy as np
from numpy.random import random

from model3D.model import SpinModel, coord_to_vector
from model3D.shell import Shell
from model3D.vectors import PyomoVector, Spin, Z_UNIT, HEX_BASIS_VECTORS,\
            ZERO_VECTOR
from model3D.grid import GridCoord


def skyrmion_model(grid_dimen):
    '''Returns a model for a two-dimensional grid containing Skyrmions, with
    dimensions given by the tuple grid_dimen'''
    param_array_list = []  # List to store shells around each spin

    #  Creates list of shells for parameters
    for i in range(grid_dimen[0]):
        i_row = []
        for j in range(grid_dimen[1]):
            #  Shell for coordinate (i, j, k)
            shell = Shell(GridCoord(i, j, 0),
                          initial_spin=Spin(random() * 2 * np.pi,
                                            random() * np.pi),
                          mag_vector=1.1 * Z_UNIT,
                          aniso_term=(0.5, Z_UNIT))

            #  Vector of the position of the spin
            pos_vector = coord_to_vector(HEX_BASIS_VECTORS, GridCoord(i, j, 0))

            #  List of neighbours in the plane
            nbr_list = [GridCoord(i - 1, j, 0), GridCoord(i + 1, j, 0),
                        GridCoord(i, j - 1, 0), GridCoord(i, j + 1, 0)]

            #  Adds the coupling with the neighbouring spins
            for coord in nbr_list:
                nbr_vector = coord_to_vector(HEX_BASIS_VECTORS, coord)
                unit_path_vector = (nbr_vector - pos_vector).normalise(1)
                dm_vector = Z_UNIT.vector_product(unit_path_vector)
                shell.add_coupling(coord, spin_coup_term=1.0,
                                   dm_vector=dm_vector)

            i_row.append([shell])
        param_array_list.append(i_row)

    #  Converts to a numpy array
    param_array = np.array(param_array_list)

    return SpinModel(param_array, basis_vectors=HEX_BASIS_VECTORS)


def febo3_model(grid_dimen, mag_vector=ZERO_VECTOR):
    '''Returns a model for a three-dimensional lattice of FeBO_3, with
    dimensions given by the tuple grid_dimen, and a magnetic field vector,
    mag_vector'''
    basis_vectors = (HEX_BASIS_VECTORS[0], HEX_BASIS_VECTORS[1],
                     (1 / 3) * HEX_BASIS_VECTORS[0] +
                     (2 / 3) * HEX_BASIS_VECTORS[1] +
                     (1 / 6) * Z_UNIT)
    param_array_list = []  # List to store shells around each spin

    # Initial spin vectors
    init_spin_1 = Spin(random() * 2 * np.pi, random() * np.pi)
    init_spin_2 = Spin(random() * 2 * np.pi, random() * np.pi)

    #  Creates list of shells for parameters
    for i in range(grid_dimen[0]):
        i_row = []
        for j in range(grid_dimen[1]):
            j_row = []
            for k in range(grid_dimen[2]):
                shell = None
                if j < grid_dimen[1] % 2:
                    shell = Shell(GridCoord(i, j, k),
                                  initial_spin=init_spin_1,
                                  aniso_term=(-100, Z_UNIT),
                                  mag_vector=mag_vector)
                else:
                    shell = Shell(GridCoord(i, j, k),
                                  initial_spin=init_spin_2,
                                  aniso_term=(-100, Z_UNIT),
                                  mag_vector=mag_vector)

                #  Coupling for spins below
                shell.add_coupling(GridCoord(i, j, k - 1),
                                   spin_coup_term=-10.3,
                                   dm_vector=PyomoVector(cart_comp=(0, 0,
                                                                    -0.5)))
                shell.add_coupling(GridCoord(i + 1, j, k - 1),
                                   spin_coup_term=-10.3,
                                   dm_vector=PyomoVector(cart_comp=(0, 0,
                                                                    -0.5)))
                shell.add_coupling(GridCoord(i, j + 1, k - 1),
                                   spin_coup_term=-10.3,
                                   dm_vector=PyomoVector(cart_comp=(0, 0,
                                                                    -0.5)))

                #  Coupling for spins above
                shell.add_coupling(GridCoord(i, j, k + 1),
                                   spin_coup_term=-10.3,
                                   dm_vector=-PyomoVector(cart_comp=(0, 0,
                                                                     -0.5)))
                shell.add_coupling(GridCoord(i - 1, j, k + 1),
                                   spin_coup_term=-10.3,
                                   dm_vector=-PyomoVector(cart_comp=(0, 0,
                                                                     -0.5)))
                shell.add_coupling(GridCoord(i, j - 1, k + 1),
                                   spin_coup_term=-10.3,
                                   dm_vector=-PyomoVector(cart_comp=(0, 0,
                                                                     -0.5)))
                j_row.append(shell)
            i_row.append(j_row)
        param_array_list.append(i_row)

    #  Converts to a numpy array
    param_array = np.array(param_array_list)

    return SpinModel(param_array, basis_vectors=basis_vectors)


def simple_cubic_model(grid_dimen, spin_coup_term=1.0, mag_vector=ZERO_VECTOR):
    '''Returns a model for a three-dimensional, cubic lattice, with spins
    interacting with their closest neighbours by the spin coupling term, and an
    applied magnetic field given by mag_vector'''
    param_array_list = []  # List to store shells around each spin

    #  Creates list of shells for parameters
    for i in range(grid_dimen[0]):
        i_row = []
        for j in range(grid_dimen[1]):
            j_row = []
            for k in range(grid_dimen[2]):
                #  Shell for coordinate (i, j, k)
                shell = Shell(GridCoord(i, j, k),
                              initial_spin=Spin(0,
                                                0),
                              mag_vector=mag_vector)

                #  Adds coupling with adjacent neighbours
                for coord in GridCoord(i, j, k).adjacent_nbrs():
                    shell.add_coupling(coord, spin_coup_term=spin_coup_term)

                j_row.append(shell)
            i_row.append(j_row)
        param_array_list.append(i_row)

    # Converts to a numpy array
    param_array = np.array(param_array_list)

    return SpinModel(param_array)
