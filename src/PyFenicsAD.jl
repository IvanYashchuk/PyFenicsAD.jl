module PyFenicsAD

import Zygote
using PyCall

fenics_numpy = pyimport("fenics_numpy")
fem_eval = fenics_numpy.fem_eval
vjp_fem_eval = fenics_numpy.vjp_fem_eval
fenics_to_numpy = fenics_numpy.fenics_to_numpy
numpy_to_fenics = fenics_numpy.numpy_to_fenics

export fem_eval, vjp_fem_eval, fenics_to_numpy, numpy_to_fenics

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
