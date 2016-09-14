# SpinModel3D
A Python package for modelling a three-dimensional lattice of interacting spins. Can be used to model many magnetic materials, examples that have been modelled so far include ferromagnets, anti-ferromagnets, the weak-ferromagnet FeBO<sub>3</sub> and skyrmions.

The package uses mathematical optimisation software to create the models, expressing the physics problem in terms of an objective function to be minimised. The model of the spins follows closely to what the Classical Heisenberg model does.

The documentation for the code is contained in the "" file.

## Dependencies
* Python 2.7.
* A scientific distribution of Python (such as Anaconda) is recommended, else you will have to manually install NumPy and matplotlib.
* Pyomo: This is the optimisation package used. Installation instructions are on the [Pyomo website](http://www.pyomo.org/installation/).
* Ipopt: This is the solver used by Pyomo. Installation instructions are on the [Ipopt website](http://www.coin-or.org/Ipopt/documentation/node10.html)

## Installation
Clone the repository. The package is the folder model3D, so ensure you have the location in your Python path in order to import from it. 

## Using the code
The documentation provides descriptions on how to use the code.

## Acknowledgments
* The Pyomo modelling language
* The Ipopt solver
* Diamond Light Source Ltd, where the code was written during the summer of 2016



