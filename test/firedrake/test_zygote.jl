using PyFenicsAD
using FiniteDifferences
using Zygote
using PyCall
using Test

firedrake = pyimport("firedrake")
firedrake_adjoint = pyimport("firedrake_adjoint")
ufl = pyimport("ufl")

mesh = firedrake.UnitSquareMesh(3, 2)
V = firedrake.FunctionSpace(mesh, "P", 1)

function assemble_firedrake(u, kappa0, kappa1)

    x = firedrake.SpatialCoordinate(mesh)
    f = x[1]

    inner, grad, dx = ufl.inner, ufl.grad, ufl.dx
    J_form = 0.5 * inner(kappa0 * grad(u), grad(u)) * dx - kappa1 * f * u * dx
    # J = firedrake.assemble(J_form)
    # Avoid PyCall's automatic conversion to Julia types
    J = pycall(firedrake.assemble, PyObject, J_form)
    return J
end

templates = (firedrake.Function(V), firedrake.Constant(0.0), firedrake.Constant(0.0))
inputs = (ones(V.dim()), ones(1) * 0.5, ones(1) * 0.6)

zygote_assemble_firedrake(inputs...) =
    evaluate_primal(assemble_firedrake, templates, inputs...)[1]

Zygote.@adjoint function zygote_assemble_firedrake(inputs...)
    pyout = pycall(evaluate_primal, PyObject, assemble_firedrake, templates, inputs...)
    numpy_output, firedrake_output, firedrake_inputs, tape =
        [get(pyout, PyObject, i) for i = 0:3]

    function pullback_fun(g)
        vjp_out = evaluate_pullback(firedrake_output, firedrake_inputs, tape, g)
    end

    return get(pyout, 0), pullback_fun
end

hh(inputs...) = zygote_assemble_firedrake(inputs...)
hh0(x) = hh(x, inputs[2], inputs[3])
hh1(y) = hh(inputs[1], y, inputs[3])
hh2(z) = hh(inputs[1], inputs[2], z)

@testset "zygote_assemble_forward" begin
    out = zygote_assemble_firedrake(inputs...)

    u1 = firedrake.interpolate(firedrake.Constant(1.0), V)
    J = assemble_firedrake(u1, firedrake.Constant(0.5), firedrake.Constant(0.6))
    @test out[1] == PyCall.convert(Float64, J)
end

@testset "zygote_assemble_vjp" begin
    out, pullback_fun = Zygote.pullback(hh, inputs...)

    g = ones(size(out))
    vjp_out = pullback_fun(g)

    fdm = FiniteDifferences.central_fdm(2, 1)

    fdm_jac1 = FiniteDifferences.jacobian(fdm, hh0, inputs[1])
    fdm_jac2 = FiniteDifferences.jacobian(fdm, hh1, inputs[2])
    fdm_jac3 = FiniteDifferences.jacobian(fdm, hh2, inputs[3])

    @test isapprox(vjp_out[1], fdm_jac1[1]', atol = 1e-5)
    @test isapprox(vjp_out[2], fdm_jac2[1]', atol = 1e-5)
    @test isapprox(vjp_out[3], fdm_jac3[1]', atol = 1e-5)
end

mesh = firedrake.UnitSquareMesh(6, 5)
V = firedrake.FunctionSpace(mesh, "P", 1)

function solve_firedrake(kappa0, kappa1)

    x = firedrake.SpatialCoordinate(mesh)
    f = x[1]

    u = firedrake.Function(V)
    bcs = [firedrake.DirichletBC(V, firedrake.Constant(0.0), "on_boundary")]

    inner, grad, dx = ufl.inner, ufl.grad, ufl.dx
    JJ = 0.5 * inner(kappa0 * grad(u), grad(u)) * dx - kappa1 * f * u * dx
    v = firedrake.TestFunction(V)
    F = firedrake.derivative(JJ, u, v)
    firedrake.solve(F == 0, u, bcs = bcs)
    return u
end


templates = (firedrake.Constant(0.0), firedrake.Constant(0.0))
inputs = (ones(1) * 0.5, ones(1) * 0.6)

zygote_solve_firedrake(inputs...) =
    evaluate_primal(solve_firedrake, templates, inputs...)[1]

Zygote.@adjoint function zygote_solve_firedrake(inputs...)
    pyout = pycall(evaluate_primal, PyObject, solve_firedrake, templates, inputs...)
    numpy_output, firedrake_output, firedrake_inputs, tape =
        [get(pyout, PyObject, i) for i = 0:3]

    function pullback_fun(g)
        vjp_out = vjp_out = evaluate_pullback(firedrake_output, firedrake_inputs, tape, g)
    end

    return get(pyout, 0), pullback_fun
end

@testset "zygote_solve_forward" begin
    out = zygote_solve_firedrake(inputs...)
    u = solve_firedrake(firedrake.Constant(0.5), firedrake.Constant(0.6))
    @test isapprox(out, to_numpy(u))
end

@testset "zygote_solve_pullback" begin
    out, pullback_fun = Zygote.pullback(zygote_solve_firedrake, inputs...)
    # g = np.ones_like(numpy_output)
    g = ones(size(out))
    vjp_out = pullback_fun(g)
    @test isapprox(vjp_out[1], [-1.13533303])
    @test isapprox(vjp_out[2], [0.94611086])
end
