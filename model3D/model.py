'Model for three dimensional spins on a three dimensional lattice'

from __future__ import division

import os
from pyomo.environ import Var, Objective, value, ConcreteModel, RangeSet, \
                          Reals, minimize
from pyomo.opt import SolverFactory
import numpy as np

from model3D.vectors import PyomoVector, Spin, CUBIC_BASIS_VECTORS
from model3D.grid import GridCoord
from model3D.plotting import plot_spin_model, SpinColors


def _init_rule_azi(pyomo_model, i, j, k):
    'Initialisation rule for the azimuth angles'
    spin = pyomo_model.param_array[value(i), value(j), value(k)].initial_spin
    return value(spin.azi)


def _init_rule_pol(pyomo_model, i, j, k):
    'Initialisation rule for the polar angles'
    spin = pyomo_model.param_array[value(i), value(j), value(k)].initial_spin
    return value(spin.pol)


def _get_bound_coord(pyomo_model, input_coord):
    '''Returns the coordinate corresponding to the boundary condition for
     the input coordinate'''
    if input_coord in pyomo_model.coord_array:
        # If the coordinate is in the grid, return the coordinate
        return input_coord
    elif pyomo_model.boundary == 'zero':
        return 'zero'  # Returning 'zero' indicates there is "no spin" here
    elif pyomo_model.boundary == 'periodic':
        # Puts required data for coordinate, points and length of the grid in
        # lists in order to loop over them
        component_list = [input_coord.i, input_coord.j, input_coord.k]
        grid_points_list = [pyomo_model.grid_i_points,
                            pyomo_model.grid_j_points,
                            pyomo_model.grid_k_points]
        grid_no_list = [pyomo_model.grid_i_no,
                        pyomo_model.grid_j_no,
                        pyomo_model.grid_k_no]

        # Finds the corresponding grid point for all three coordinates
        for p in range(3):
            if value(component_list[p]) < 0:
                while not component_list[p] in grid_points_list[p]:
                    component_list[p] += grid_no_list[p]
            elif value(component_list[p]) >= grid_no_list[p]:
                while not component_list[p] in grid_points_list[p]:
                    component_list[p] -= grid_no_list[p]

        return GridCoord(component_list[0],
                         component_list[1],
                         component_list[2])


def _get_spin(pyomo_model, input_coord):
    'Returns the spin corresponding to the position given by input_coord'
    bound_coord = _get_bound_coord(pyomo_model, input_coord)
    if bound_coord == 'zero':
        return PyomoVector(cart_comp=(0.0, 0.0, 0.0))
    else:
        return Spin(pyomo_model.azi[bound_coord.i,
                                    bound_coord.j,
                                    bound_coord.k],
                    pyomo_model.pol[bound_coord.i,
                                    bound_coord.j,
                                    bound_coord.k])


def _nbr_coup_sum(pyomo_model, input_coord):
    '''Returns the sum of the interactions the spin at input_coord has with its
    neighbours that are listed in its shell, including the spin coupling and
    Dzyaloshinskii-Moriya exchange'''
    input_spin = _get_spin(pyomo_model, input_coord)
    param_shell = pyomo_model.param_array[value(input_coord.i),
                                          value(input_coord.j),
                                          value(input_coord.k)]

    nbr_spin_list = []  # List to store the neighbouring spins
    coup_const_list = []  # List to store the spin coupling constants
    dm_vector_list = []  # List to store the DM vectors for each of the spins

    for nbr_coord in param_shell.coord_list:
        nbr_spin_list.append(_get_spin(pyomo_model, nbr_coord))
        coup_const_list.append(param_shell.get_spin_coup(nbr_coord))
        dm_vector_list.append(param_shell.get_dm_vector(nbr_coord))

    return sum(coup_const_list[n] *
               input_spin.scalar_product(nbr_spin_list[n]) +
               dm_vector_list[n].scalar_product(
                   input_spin.vector_product(nbr_spin_list[n]))
               for n in range(len(nbr_spin_list)))


def _hamiltonian(pyomo_model):
    'The Hamiltonian function for the given model to be minimized by Pyomo'
    total_nbr_coup = 0.0  # Contribution from spin coupling with neighbours
    total_mag = 0.0  # Contribution from coupling of magnetic field and spins
    total_aniso = 0.0  # Contribution from coupling with the anisotropy
    for coord in pyomo_model.coord_array:
        # Spin at the input coordinate location
        input_spin = _get_spin(pyomo_model, coord)

        # Shell for coupling parameters
        param_shell = pyomo_model.param_array[value(coord.i),
                                              value(coord.j),
                                              value(coord.k)]

        # Adds sum of scalar products of nearest neighbours
        total_nbr_coup += _nbr_coup_sum(pyomo_model, coord)

        # Adds scalar product of the spin with the magnetic field
        mag_vector = param_shell.mag_vector
        total_mag += input_spin.scalar_product(mag_vector)

        # Adds the coupling with the anisotropy
        total_aniso += param_shell.aniso_strength \
            * (input_spin.scalar_product(param_shell.aniso_direction)) ** 2

    return -total_nbr_coup - total_mag - total_aniso


def coord_to_vector(basis_vectors, coord):
    'Returns the vector that the GridCoord coord represents in real space'
    return coord.i * basis_vectors[0]\
        + coord.j * basis_vectors[1]\
        + coord.k * basis_vectors[2]


def save_model_results(parent_dir, save_dir_name, spins_list):
    '''Saves the model in the parent directory, parent_dir. Results list is a
    list in the format of the output of SpinModel.get_model_results'''
    # Creates a directory to save the results in
    save_dir = parent_dir + '/' + save_dir_name
    if not os.path.exists(parent_dir + '/' + save_dir_name):
        os.mkdir(parent_dir + '/' + save_dir_name)

    # Gets the results to save into the directory
    xyz_list = ['x', 'y', 'z']  # Characters used for saving file names
    for m in range(3):
        np.save(save_dir + '/' + xyz_list[m] + '_pos',
                spins_list[0][m])
        np.save(save_dir + '/' + xyz_list[m] + '_spin_comps',
                spins_list[1][m])


class SpinModel(object):
    'Class for creating the model of a 3D lattice of 3D spins'
    def __init__(self, boundary, param_array,
                 basis_vectors=CUBIC_BASIS_VECTORS):
        '''
        Input parameters required to create the model:
        * boundary - String for the boundary condition, either "periodic" or
          "zero"
        * param_array - A numpy array of shells, which will have one shell per
          spin on the grid, and the shell is a Shell object which stores the
          parameters needed for coupling that particular spin
        * basis_vectors - The basis vectors of the lattice
        '''
        self.basis_vectors = basis_vectors

        # Defining the Pyomo model
        self.model = ConcreteModel()

        # =====================================================================
        # Storing parameters as model objects
        # =====================================================================
        self.model.boundary = boundary

        self.model.param_array = param_array

        # =====================================================================
        # Derived objects from the parameters
        # =====================================================================
        # The number of points in the i, j and k directions in grid space
        self.model.grid_i_no = param_array.shape[0]
        self.model.grid_j_no = param_array.shape[1]
        self.model.grid_k_no = param_array.shape[2]

        # Creates an array of all the coordinates in the grid
        self.model.coord_array = np.array([param_shell.centre
                                           for param_shell
                                           in param_array.flatten()])

        # Ranges for the grid points
        self.model.grid_i_points = RangeSet(0, self.model.grid_i_no - 1)
        self.model.grid_j_points = RangeSet(0, self.model.grid_j_no - 1)
        self.model.grid_k_points = RangeSet(0, self.model.grid_k_no - 1)

        # Variables for the azimuth and polar angles of the spins
        self.model.azi = Var(self.model.grid_i_points,
                             self.model.grid_j_points,
                             self.model.grid_k_points,
                             domain=Reals,
                             bounds=(0.0, 2 * np.pi),
                             initialize=_init_rule_azi)

        self.model.pol = Var(self.model.grid_i_points,
                             self.model.grid_j_points,
                             self.model.grid_k_points,
                             domain=Reals,
                             bounds=(0.0, np.pi),
                             initialize=_init_rule_pol)

        # Objective function to minimize
        self.model.OBJ = Objective(rule=_hamiltonian, sense=minimize)

        # Creates the results variables
        self.results_array_azi = np.zeros((value(self.model.grid_i_no),
                                           value(self.model.grid_j_no),
                                           value(self.model.grid_k_no)))
        self.results_array_pol = np.copy(self.results_array_azi)
        self.results_obj = 0.0

    def solve(self):
        'Solves the given problem using the ipopt solver'
        # Solves the model
        opt = SolverFactory("ipopt")
        opt.solve(self.model)

        for input_coord in self.model.coord_array:
            i, j, k = input_coord.i, input_coord.j, input_coord.k
            self.results_array_azi[i, j, k] = value(self.model.azi[i, j, k])
            self.results_array_pol[i, j, k] = value(self.model.pol[i, j, k])

        self.results_obj = value(self.model.OBJ)

    def get_model_results(self):
        'Returns arrays containing the positions and components of the spins'
        coord_array = self.model.coord_array  # To reduce length of lines

        # Arrays to store x, y and z positions of the spins
        x_pos, y_pos, z_pos = [], [], []
        for coord in coord_array:
            # Converts coord into vector in real space
            pos_vector = coord_to_vector(self.basis_vectors, coord)
            x_pos.append(pos_vector.x)
            y_pos.append(pos_vector.y)
            z_pos.append(pos_vector.z)

        # Converts the lists into numpy arrays
        x_pos = np.array(x_pos)
        y_pos = np.array(y_pos)
        z_pos = np.array(z_pos)

        # Components of the vectors of the spins
        x_spin_comps = np.zeros(x_pos.size)
        y_spin_comps = np.copy(x_spin_comps)
        z_spin_comps = np.copy(x_spin_comps)

        # Fills the spin component arrays
        for index, coord in enumerate(coord_array):
            spin = _get_spin(self.model, coord)
            x_spin_comps[index] = value(spin.x)
            y_spin_comps[index] = value(spin.y)
            z_spin_comps[index] = value(spin.z)

        return ((x_pos, y_pos, z_pos),
                (x_spin_comps, y_spin_comps, z_spin_comps))

    def plot(self, spin_colors=SpinColors.plain, plot_points=True,
             x_limits=(-np.inf, np.inf), y_limits=(-np.inf, np.inf),
             z_limits=(-np.inf, np.inf)):
        'Passes the required data to lattice_3D_plot for plotting'
        spin_results = self.get_model_results()
        plot_spin_model(spin_results[0], spin_results[1],
                        spin_colors=spin_colors, plot_points=plot_points,
                        x_limits=x_limits, y_limits=y_limits,
                        z_limits=z_limits)
