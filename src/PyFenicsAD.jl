module PyFenicsAD

import ChainRulesCore
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

"""
A convenience macro that creates a julia function and registers forward or reverse rules for the provided FEniCS/Firedrake `fem_function`.
"""
macro register_fem_function(julia_function_name, fem_templates, fem_function)
    name_str = "$julia_function_name"
    quote
        function $(esc(julia_function_name))(inputs...)
            return evaluate_primal($(esc(fem_function)), $(esc(fem_templates)), inputs...)[1]
        end

        function ChainRulesCore.rrule(::typeof($(esc(julia_function_name))), inputs...)
            pyout = pycall(
                evaluate_primal,
                PyObject,
                $(esc(fem_function)),
                $(esc(fem_templates)),
                inputs...,
            )
            numpy_output, fem_output, fem_inputs, tape =
                [get(pyout, PyObject, i) for i = 0:3]

            # Only single-output functions are supported for now
            function fem_pullback(g)
                vjp_out = evaluate_pullback(fem_output, fem_inputs, tape, g)
                return (ChainRulesCore.NO_FIELDS, vjp_out...)
            end
            return get(pyout, 0), fem_pullback
        end

        function ChainRulesCore.frule(
            (_, Δinputs),
            ::typeof($(esc(julia_function_name))),
            inputs...,
        )
            pyout = pycall(
                evaluate_primal,
                PyObject,
                $(esc(fem_function)),
                $(esc(fem_templates)),
                inputs...,
            )
            numpy_output, fem_output, fem_inputs, tape =
                [get(pyout, PyObject, i) for i = 0:3]

            ∂numpy_output = evaluate_pushforward(fem_output, fem_inputs, tape, Δinputs)
            return get(pyout, 0), ∂numpy_output
        end

        println("Registered FEniCS/Firedrake function with name ", $name_str, "!")
    end
end

export @register_fem_function

end # module
