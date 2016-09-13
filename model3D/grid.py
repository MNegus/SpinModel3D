'Contains the class for 3D grid coordinates'


class GridCoord(object):
    'Class for coordinates of the points of the grid where spins are.'
    def __init__(self, i, j, k):
        self.i = i
        self.j = j
        self.k = k

    def l1_distance(self, grid_coord):
        '''Calculates the distance between itself and another grid_coord
        according to the L1 metric'''
        return abs(self.i - grid_coord.i)\
            + abs(self.j - grid_coord.j)\
            + abs(self.k - grid_coord.k)

    def to_tuple(self):
        'Converts to a tuple'
        return (self.i, self.j, self.k)

    def __eq__(self, grid_coord):
        'Overloads the == function to compare coordinates'
        if isinstance(grid_coord, GridCoord):
            return self.is_equal(grid_coord)
        return NotImplemented

    def is_equal(self, grid_coord):
        'Prints True if self is the same coordinate as grid_coord'
        return self.i == grid_coord.i and self.j == grid_coord.j \
            and self.k == grid_coord.k

    def adjacent_nbrs(self):
        'Returns the GridCoords of the adjacent neighbours to grid_coord'
        return (GridCoord(self.i - 1, self.j, self.k),
                GridCoord(self.i + 1, self.j, self.k),
                GridCoord(self.i, self.j - 1, self.k),
                GridCoord(self.i, self.j + 1, self.k),
                GridCoord(self.i, self.j, self.k - 1),
                GridCoord(self.i, self.j, self.k + 1))


class CoordList(list):
    '''Class for a list of grid coordinates, overriding some methods'''
    def in_list(self, coord):
        '''Returns True if the GridCoord object coord is in the list'''
        for item in self:
            if coord == item:
                return True
        return False

    def __contains__(self, item):
        if isinstance(item, GridCoord):
            return self.in_list(item)
        else:
            return list.__contains__(self, item)

    def append(self, item):
        '''Appends an item to a list only if it isn't already in it'''
        if item in self:
            raise Exception("Item already in list")
        else:
            super(CoordList, self).append(item)

    def index(self, item):
        '''Returns the index in the list which the item is stored'''
        for i, element in enumerate(self):
            if element == item:
                return i
        raise ValueError(str(item) + ' is not in list')
