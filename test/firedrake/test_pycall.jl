using PyCall, Test, FiniteDifferences

firedrake = pyimport("firedrake")
firedrake_adjoint = pyimport("firedrake_adjoint")
pyadjoint = pyimport("pyadjoint")
ufl = pyimport("ufl")
np = pyimport("numpy")

fecr = pyimport("fecr")
evaluate_primal = fecr.evaluate_primal
evaluate_pullback = fecr.evaluate_pullback
to_numpy = fecr.to_numpy


mesh = firedrake.UnitSquareMesh(3, 2)
V = firedrake.FunctionSpace(mesh, "P", 1)

function assemble_firedrake(u, kappa0, kappa1)

    x = firedrake.SpatialCoordinate(mesh)
    f = x[0]

    inner, grad, dx = ufl.inner, ufl.grad, ufl.dx
    J_form = 0.5 * inner(kappa0 * grad(u), grad(u)) * dx - kappa1 * f * u * dx
    # J = firedrake.assemble(J_form)
    # Avoid PyCall's automatic conversion to Julia types
    J = pycall(firedrake.assemble, PyObject, J_form)
    return J
end

templates = (firedrake.Function(V), firedrake.Constant(0.0), firedrake.Constant(0.0))
inputs = (ones(V.dim()), ones(1) * 0.5, ones(1) * 0.6)

hh(args...) = evaluate_primal(assemble_firedrake, templates, args...)[1]
hh0(x) = hh(x, inputs[2], inputs[3])
hh1(y) = hh(inputs[1], y, inputs[3])
hh2(z) = hh(inputs[1], inputs[2], z)

@testset "assemble_forward" begin
    pyout = pycall(evaluate_primal, PyObject, assemble_firedrake, templates, inputs...)
    numpy_output, firedrake_output, firedrake_inputs, tape =
        [get(pyout, PyObject, i) for i = 0:3]
    @test pybuiltin(:isinstance)(firedrake_output, pyadjoint.AdjFloat)

    u1 = firedrake.interpolate(firedrake.Constant(1.0), V)
    J = assemble_firedrake(u1, firedrake.Constant(0.5), firedrake.Constant(0.6))
    @test np.isclose(numpy_output, J)

    jnumpy_output = PyCall.convert(Float64, numpy_output)
    jJ = PyCall.convert(Float64, J)
    @test isapprox(jnumpy_output, jJ)
end

@testset "assemble_vjp" begin
    pyout = pycall(evaluate_primal, PyObject, assemble_firedrake, templates, inputs...)
    numpy_output, firedrake_output, firedrake_inputs, tape =
        [get(pyout, PyObject, i) for i = 0:3]

    g = np.ones_like(numpy_output)
    vjp_out = evaluate_pullback(firedrake_output, firedrake_inputs, tape, g)

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
    f = x[0]

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

@testset "solve_forward" begin
    numpy_output, _, _, _ = evaluate_primal(solve_firedrake, templates, inputs...)
    u = solve_firedrake(firedrake.Constant(0.5), firedrake.Constant(0.6))
    @test isapprox(numpy_output, to_numpy(u))
end

@testset "solve_vjp" begin
    numpy_output, firedrake_output, firedrake_inputs, tape =
        evaluate_primal(solve_firedrake, templates, inputs...)
    # g = np.ones_like(numpy_output)
    g = ones(size(numpy_output))
    vjp_out = evaluate_pullback(firedrake_output, firedrake_inputs, tape, g)
    @test isapprox(vjp_out[1], [-2.91792642])
    @test isapprox(vjp_out[2], [2.43160535])
end
