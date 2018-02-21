# Optimize Arguments

Minimization of least-squares distance between non-linear functions via optimization of free parameters (arguments).

In problems such as those of curve fitting and dynamical scaling, we often need to evaluate free parameters that will minimize the distance between different non-linear functions of some independent variable(s). The `optimize_arguments` function achieves this by performing an iterative grid-search over the space of free parameters (arguments) as specified by the user.


## Getting Started

### Prerequisites

- [Anaconda Python 2.7.x or 3.x.x](https://www.continuum.io/downloads)

### Installing

Installation does not require root access. After downloading the repository and extracting the files in it, run the following command from a terminal to install:
 ```
 python setup.py install --user
 ```

### Using The Tool

Please refer to the [Examples](Examples) folder for example guides (Jupyter Notebooks) of how to use `optimize_arguments` to infer optimal parameters.

## Authors
**Harsh Chaturvedi**


## License

This project is licensed under the [MIT License](LICENSE.txt).