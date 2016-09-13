'''Class for storing the coupling parameters of each spin with its
neighbours, stored in shells'''

from model3D.grid import CoordList
from model3D.vectors import PyomoVector, Spin, ZERO_VECTOR


class Shell(object):
    '''Class for a shell around a grid point storing the coupling parameters of
    the spin at that point with its neighbouring spins'''
    def __init__(self, centre_coord, initial_spin=Spin(0.0, 0.0),
                 mag_vector=ZERO_VECTOR, aniso_term=(0.0, ZERO_VECTOR)):
        '''The coordinate of the centre of the shell is given'''
        self.centre = centre_coord  # Grid coord of the centre of the shell
        self.initial_spin = initial_spin
        self.mag_vector = mag_vector  # Magnetic field vector at the centre

        self.aniso_strength = aniso_term[0]
        self.aniso_direction = aniso_term[1].normalise(1.0)

        self.coord_list = CoordList([])  # List of the coordinates for coupling
        self.spin_coup_dict = {}  # Dict for values of spin coupling params
        self.dm_vector_dict = {}  # Dict for the DM-vector when coupling with
                                  # each coord

    def add_coupling(self, coord, spin_coup_term=0.0, dm_vector=ZERO_VECTOR):
        '''Adds a coordinate of a spin and coupling terms for the centre spin
        to interact with'''
        if coord in self.coord_list:
            for key_coord in self.spin_coup_dict.keys():
                if key_coord == coord:
                    self.spin_coup_dict[key_coord] = spin_coup_term

            for key_coord in self.dm_vector_dict.keys():
                if key_coord == coord:
                    self.dm_vector_dict[key_coord] = dm_vector
        else:
            self.coord_list.append(coord)
            self.spin_coup_dict[coord] = spin_coup_term
            self.dm_vector_dict[coord] = dm_vector

    def get_spin_coup(self, coord):
        '''Returns the spin coupling term at coord'''
        coord_index = self.coord_list.index(coord)
        return self.spin_coup_dict[self.coord_list[coord_index]]

    def get_dm_vector(self, coord):
        '''Returns the Dzyaloshinskii-Moriya interaction vector at coord'''
        coord_index = self.coord_list.index(coord)
        return self.dm_vector_dict[self.coord_list[coord_index]]
