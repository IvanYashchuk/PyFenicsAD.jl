module PyFenicsAD

# import Zygote
using PyCall

const fecr = PyNULL()
const to_numpy = PyNULL()
const from_numpy = PyNULL()
const evaluate_primal = PyNULL()
const evaluate_pullback = PyNULL()
const evaluate_pushforward = PyNULL()

function __init__()
    copy!(fecr, pyimport("fecr"))
    copy!(to_numpy, fecr.to_numpy)
    copy!(from_numpy, fecr.from_numpy)
    copy!(evaluate_primal, fecr.evaluate_primal)
    copy!(evaluate_pullback, fecr.evaluate_pullback)
    copy!(evaluate_pushforward, fecr.evaluate_pushforward)
end

export fecr, to_numpy, from_numpy, evaluate_primal, evaluate_pullback, evaluate_pushforward

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
