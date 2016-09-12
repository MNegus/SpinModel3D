'Class for a Pyomo vector and a Spin object which is derived from PyomoVector'

from __future__ import division

from pyomo.environ import sqrt, value, cos, sin, acos, atan

import numpy as np


class PyomoVector(object):
    '''Generic vector object which stores both the Cartesian and Spherical
    components of the vector'''
    def __init__(self, cart_comp=None, sph_comp=None):
        '''Initialised the vector either in Cartesian components or Spherical
        components. If both are given, only looks at cart_comp'''
        if cart_comp is not None:
            self.x = cart_comp[0]
            self.y = cart_comp[1]
            self.z = cart_comp[2]

            self.length = sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)
            if value(self.length) == 0:
                self.azi = 0.0
                self.pol = 0.0
            else:
                self.pol = acos(self.z / self.length)
                if value(self.x) != 0:
                    self.azi = atan(self.y / self.x)
                else:
                    self.azi = 0.0
        elif sph_comp is not None:
            self.length = sph_comp[0]
            self.azi = sph_comp[1]
            self.pol = sph_comp[2]

            self.x = self.length * cos(self.azi) * sin(self.pol)
            self.y = self.length * sin(self.azi) * sin(self.pol)
            self.z = self.length * cos(self.pol)
        else:
            raise Exception('''Must give either Cartesian or Spherical
                            components''')

    def _config_sph(self):
        'Configures the spherical components from the Cartesian'
        self.length = sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)
        if value(self.length) == 0:
            self.azi = 0.0
            self.pol = 0.0
        else:
            self.pol = acos(self.z / self.length)
            if value(self.x) != 0:
                self.azi = atan(self.y / self.x)
            else:
                self.azi = 0.0

    def _config_cart(self):
        'Configures the Cartesian components from the spherical'
        self.x = self.length * cos(self.azi) * sin(self.pol)
        self.y = self.length * sin(self.azi) * sin(self.pol)
        self.z = self.length * cos(self.pol)

    def set_x(self, comp_value):
        'Sets the x-component to the specified value'
        self.x = comp_value
        self._config_sph()

    def set_y(self, comp_value):
        'Sets the y-component to the specified value'
        self.y = comp_value
        self._config_sph()

    def set_z(self, comp_value):
        'Sets the z-component to the specified value'
        self.z = comp_value
        self._config_sph()

    def set_length(self, comp_value):
        'Sets the length component to the specified value'
        self.length = comp_value
        self._config_cart()

    def set_azi(self, comp_value):
        'Sets the azi component to the specified value'
        self.azi = comp_value
        self._config_cart()

    def set_pol(self, comp_value):
        'Sets the pol component to the specified value'
        self.pol = comp_value
        self._config_cart()

    def add(self, vector):
        'Returns the addition of itself with another PyomoVector'
        return PyomoVector(cart_comp=(self.x + vector.x,
                                      self.y + vector.y,
                                      self.z + vector.z))

    def __add__(self, vector):
        'Overrides the addition operator'
        if isinstance(vector, PyomoVector):
            return self.add(vector)
        return NotImplemented

    def multiply(self, scalar):
        'Returns the result of the vector being multiplied by a scalar'
        return PyomoVector(cart_comp=(scalar * self.x,
                                      scalar * self.y,
                                      scalar * self.z))

    def __mul__(self, scalar):
        'Overrides the multiplication operator for scalars'
        return self.multiply(scalar)

    def __rmul__(self, scalar):
        'Overrides the reverse multiplication for scalars'
        return self * scalar

    def __sub__(self, vector):
        'Overrides the subtraction of vectors'
        if isinstance(vector, PyomoVector):
            neg_vector = -1 * vector
            return self + neg_vector
        return NotImplemented

    def __pos_(self):
        'Overrides the "+" operand in front of a vector'
        return self

    def __neg__(self):
        'Overrides the "-" operand in front of a vector'
        return -1 * self

    def __abs__(self):
        'Overrides the abs() function'
        return self.length

    def scalar_product(self, vector):
        'Returns the scalar product of itself with another PyomoVector'
        return self.x * vector.x + self.y * vector.y + self.z * vector.z

    def vector_product(self, vector):
        'Returns the vector product of itself with another PyomoVector'
        x_comp = self.y * vector.z - self.z * vector.y
        y_comp = self.z * vector.x - self.x * vector.z
        z_comp = self.x * vector.y - self.y * vector.x
        return PyomoVector(cart_comp=(x_comp, y_comp, z_comp))

    def normalise(self, new_length):
        'Returns a vector parallel to itself with given length'
        if value(abs(self)) == 0:
            return self
        else:
            factor = new_length / abs(self)
            return self * factor

    def cart_tuple(self):
        'Returns the Cartesian components in a tuple'
        return (value(self.x), value(self.y), value(self.z))

    def sph_tuple(self):
        'Returns the Spherical components in a tuple'
        return (value(self.length), value(self.azi), value(self.pol))

    def __eq__(self, vector):
        'Overides the equals method'
        if isinstance(vector, PyomoVector):
            return self.x == vector.x and\
                self.y == vector.y and\
                self.z == vector.z
        return NotImplemented


class Spin(PyomoVector):
    '''Object for a spin, which is a unit vector and defined by an azimuth and
    polar angle.'''
    def __init__(self, azi, pol):
        'Azimuth and polar angles passed in to define the spin.'
        PyomoVector.__init__(self, sph_comp=(1.0, azi, pol))

# Unit vectors in x, y, z directions
X_UNIT = PyomoVector(cart_comp=(1.0, 0.0, 0.0))
Y_UNIT = PyomoVector(cart_comp=(0.0, 1.0, 0.0))
Z_UNIT = PyomoVector(cart_comp=(0.0, 0.0, 1.0))

# Zero vector
ZERO_VECTOR = PyomoVector(cart_comp=(0.0, 0.0, 0.0))

# Standard cubic lattice basis vectors
CUBIC_BASIS_VECTORS = (X_UNIT, Y_UNIT, Z_UNIT)

# Hexagonal lattice basis vectors
HEX_BASIS_1 = X_UNIT
HEX_BASIS_2 = PyomoVector(cart_comp=(np.cos(np.radians(120)),
                                     np.sin(np.radians(120)),
                                     0.0))
HEX_BASIS_3 = Z_UNIT
HEX_BASIS_VECTORS = (HEX_BASIS_1, HEX_BASIS_2, HEX_BASIS_3)
