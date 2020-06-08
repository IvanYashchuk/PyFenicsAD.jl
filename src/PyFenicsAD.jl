module PyFenicsAD

import Zygote
using PyCall

const fenics_numpy = PyNULL()
const fem_eval = PyNULL()
const vjp_fem_eval = PyNULL()
const fenics_to_numpy = PyNULL()
const numpy_to_fenics = PyNULL()

function __init__()
    copy!(fenics_numpy, pyimport("fenics_numpy"))
    copy!(fem_eval, fenics_numpy.fem_eval)
    copy!(vjp_fem_eval, fenics_numpy.vjp_fem_eval)
    copy!(fenics_to_numpy, fenics_numpy.fenics_to_numpy)
    copy!(numpy_to_fenics, fenics_numpy.numpy_to_fenics)
end

export fem_eval, vjp_fem_eval, fenics_to_numpy, numpy_to_fenics

# TODO: Make a macro for wrapping fenics functions
# function create_zygote_fem_eval(fenics_templates::Tuple{Vararg{PyObject}})::Function
#     function decorator(fenics_function::Function)::Function
#         zygote_fem_eval(inputs...) = fem_eval(fenics_function, fenics_templates, inputs...)[1]

#         eval(quote
#             Zygote.@adjoint function zygote_fem_eval(inputs...)
#                 pyout = pycall(fem_eval, PyObject, fenics_function, fenics_templates, inputs...)
#                 numpy_output, fenics_output, fenics_inputs, tape = [get(pyout, PyObject, i) for i in 0:3]

#                 function vjp_fun(g)
#                     vjp_out = vjp_fem_eval(g, fenics_output, fenics_inputs, tape)
#                 end

#                 return get(pyout, 0), vjp_fun
#             end
#         end)
#         return zygote_fem_eval
#     end
#     return decorator
# end

end # module
