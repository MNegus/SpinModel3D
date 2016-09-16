'Function for plotting a 3D lattice'

from enum import Enum
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from model3D.vectors import ZERO_VECTOR, value


class SpinColors(Enum):
    'Enum class for the different color schemes for the spins'
    plain = 1
    alternating = 2
    z_gradient = 3
    xy_gradient = 4


def plot_spin_model(input_pos_arrays, input_comps_arrays,
                    spin_colors=SpinColors.plain, plot_points=True,
                    x_limits=(-np.inf, np.inf), y_limits=(-np.inf, np.inf),
                    z_limits=(-np.inf, np.inf), mag_vector=ZERO_VECTOR):
    'Plots the spins of a given SpinModel object'
    # Creates figure and axes
    fig = plt.figure()
    axes = fig.add_subplot(111, projection='3d')

    # Removing the points that are not within the x, y, z limits
    no_points = len(input_pos_arrays[0])

    x_pos, y_pos, z_pos, = [], [], []
    x_spin_comps, y_spin_comps, z_spin_comps = [], [], []

    for n in range(no_points):
        if x_limits[0] <= input_pos_arrays[0][n] <= x_limits[1] and\
                y_limits[0] <= input_pos_arrays[1][n] <= y_limits[1] and\
                z_limits[0] <= input_pos_arrays[2][n] <= z_limits[1]:
            x_pos.append(input_pos_arrays[0][n])
            y_pos.append(input_pos_arrays[1][n])
            z_pos.append(input_pos_arrays[2][n])

            x_spin_comps.append(input_comps_arrays[0][n])
            y_spin_comps.append(input_comps_arrays[1][n])
            z_spin_comps.append(input_comps_arrays[2][n])

    # Converting lists into numpy arrays
    x_pos, y_pos, z_pos = np.array(x_pos), np.array(y_pos), np.array(z_pos)
    x_spin_comps, y_spin_comps, z_spin_comps = np.array(x_spin_comps),\
        np.array(y_spin_comps),\
        np.array(z_spin_comps)

    # Scatter plot for the grid points where the spins are
    if plot_points:
        axes.scatter(x_pos, y_pos, z_pos, color='blue')
        # axes.scatter(x_pos[::2], y_pos[::2], z_pos[::2], color='red')
        # axes.scatter(x_pos[1::2], y_pos[1::2], z_pos[1::2], color='green')

    # Plots the spin vectors
    if spin_colors == SpinColors.plain:
        # All the spins are plotted with the same colour'
        axes.quiver(x_pos, y_pos, z_pos,
                    x_spin_comps, y_spin_comps, z_spin_comps,
                    length=0.4, arrow_length_ratio=0.5, pivot='tail',
                    linewidth=3.0)
    elif spin_colors == SpinColors.alternating:
        # The spins have alternating colours for each lattice layer
        axes.quiver(x_pos[::2], y_pos[::2], z_pos[::2],
                    x_spin_comps[::2], y_spin_comps[::2], z_spin_comps[::2],
                    length=0.4, arrow_length_ratio=0.5, pivot='tail',
                    linewidth=3.0, color='green')
        axes.quiver(x_pos[1::2], y_pos[1::2], z_pos[1::2],
                    x_spin_comps[1::2], y_spin_comps[1::2], z_spin_comps[1::2],
                    length=0.4, arrow_length_ratio=0.5, pivot='tail',
                    linewidth=3.0)
    elif spin_colors == SpinColors.z_gradient:
        # The spins have a gradient of colours depending on their z-component
        for p, z_comp in enumerate(z_spin_comps):
            color = (0, 0.5 * (1 - z_comp), 0.5 * (1 + z_comp))
            axes.quiver(x_pos[p], y_pos[p], z_pos[p],
                        x_spin_comps[p], y_spin_comps[p], z_spin_comps[p],
                        length=0.4, arrow_length_ratio=0.5, pivot='tail',
                        linewidth=3.0, color=color)

    elif spin_colors == SpinColors.xy_gradient:
        for p in range(len(x_pos)):
            color = (0.5 * (1 - x_spin_comps[p]), 0.5 * (1 + y_spin_comps[p]),
                     0)
            axes.quiver(x_pos[p], y_pos[p], z_pos[p],
                        x_spin_comps[p], y_spin_comps[p], z_spin_comps[p],
                        length=0.4, arrow_length_ratio=0.5, pivot='tail',
                        linewidth=3.0, color=color)

    # Determines how the graph should be scaled
    max_range = np.array([x_pos.max() - x_pos.min(),
                          y_pos.max() - y_pos.min(),
                          z_pos.max() - z_pos.min()]).max() / 2.0
    axes.set_xlim(x_pos.mean() - max_range, x_pos.mean() + max_range)
    axes.set_ylim(y_pos.mean() - max_range, y_pos.mean() + max_range)
    axes.set_zlim(z_pos.mean() - max_range, z_pos.mean() + max_range)

    # Draws the magnetic field vector
    axes.quiver([x_pos.mean()], [y_pos.mean()],
                [z_pos.max() + 0.1 * max_range], [value(mag_vector.x)],
                [value(mag_vector.y)], [value(mag_vector.z)],
                length=0.5 * max_range, arrow_length_ratio=0.3, pivot='tail',
                linewidth=3.0)

    # Setting axis labels
    axes.set_xlabel('x')
    axes.set_ylabel('y')
    axes.set_zlabel('z')

    plt.show()


def plot_saved_model(save_dir, spin_colors=SpinColors.plain, plot_points=True,
                     x_limits=(-np.inf, np.inf), y_limits=(-np.inf, np.inf),
                     z_limits=(-np.inf, np.inf), mag_vector=ZERO_VECTOR):
    'Plots the results that are saved in save_dir'
    load_pos_arrays, load_comps_arrays = [], []
    # Gets the results to save into the directory
    xyz_list = ['x', 'y', 'z']  # Characters used for saving file names
    for m in range(3):
        load_pos_arrays.append(np.load(
            save_dir + '/' + xyz_list[m] + '_pos.npy'))
        load_comps_arrays.append(np.load(
            save_dir + '/' + xyz_list[m] + '_spin_comps.npy'))

    plot_spin_model(load_pos_arrays, load_comps_arrays,
                    spin_colors=spin_colors, plot_points=plot_points,
                    x_limits=x_limits, y_limits=y_limits, z_limits=z_limits,
                    mag_vector=mag_vector)
