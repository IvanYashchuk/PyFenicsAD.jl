# PyFenicsAD.jl
Automatic differentiation of FEniCS models in Julia

# Installation
First find out the path to python that has FEniCS installed.
Install NumPy interface to FEniCS adjoint

    python -m pip install git+https://github.com/IvanYashchuk/numpy-fenics-adjoint.git@master

Then in Julia run

    ENV["PYTHON"] = "path/to/fenics/python"
    import Pkg
    Pkg.build("PyCall")
    ] add https://github.com/IvanYashchuk/PyFenicsAD.jl

