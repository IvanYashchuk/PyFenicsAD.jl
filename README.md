# PyFenicsAD.jl &middot; [![Build FEniCS](https://github.com/ivanyashchuk/PyFenicsAD.jl/workflows/FEniCS/badge.svg)](https://github.com/ivanyashchuk/PyFenicsAD.jl/actions?query=workflow%3AFEniCS+branch%3Amaster) [![codecov](https://codecov.io/gh/IvanYashchuk/PyFenicsAD.jl/branch/master/graph/badge.svg?token=E2QUTNOLYP)](https://codecov.io/gh/IvanYashchuk/PyFenicsAD.jl)
Automatic differentiation of FEniCS or Firedrake models in Julia

# Installation
First find out the path to python that has FEniCS or Firedrake installed.
Install FiniteElementChainRules (FECR) package that provides NumPy interface to FEniCS and Firedrake together with pushforwards and pullbacks

    python -m pip install git+https://github.com/IvanYashchuk/fecr.git@master

Then in Julia run

    ENV["PYTHON"] = "path/to/fenics_or_firedrake/python"
    import Pkg
    Pkg.build("PyCall")
    ] add https://github.com/IvanYashchuk/PyFenicsAD.jl

