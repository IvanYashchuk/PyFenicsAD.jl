# PyFenicsAD.jl
Automatic differentiation of FEniCS models in Julia

# Installation
First find out the path to python that has FEniCS installed.
Then in Julia run

    ENV["PYTHON"] = "path/to/fenics/python"
    import Pkg
    Pkg.build("PyCall")

