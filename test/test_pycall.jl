using PyFenicsAD
using PyCall, Test, FiniteDifferences

fenics = pyimport("fenics")
fa = pyimport("fenics_adjoint")
pyadjoint = pyimport("pyadjoint")
ufl = pyimport("ufl")
np = pyimport("numpy")

fenics_numpy = pyimport("fenics_numpy")
fem_eval = fenics_numpy.fem_eval
vjp_fem_eval = fenics_numpy.vjp_fem_eval
fenics_to_numpy = fenics_numpy.fenics_to_numpy
numpy_to_fenics = fenics_numpy.numpy_to_fenics


mesh = fa.UnitSquareMesh(3, 2)
V = fenics.FunctionSpace(mesh, "P", 1)

function assemble_fenics(u, kappa0, kappa1)

    f = fa.Expression(
        "10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=2
    )

    inner, grad, dx = ufl.inner, ufl.grad, ufl.dx
    J_form = 0.5 * inner(kappa0 * grad(u), grad(u)) * dx - kappa1 * f * u * dx
    # J = fa.assemble(J_form)
    # Avoid PyCall's automatic conversion to Julia types
    J = pycall(fa.assemble, PyObject, J_form)
    return J
end

templates = (fa.Function(V), fa.Constant(0.0), fa.Constant(0.0))
inputs = (ones(V.dim()), ones(1) * 0.5, ones(1) * 0.6)

hh(args...) = fem_eval(assemble_fenics, templates, args...)[1]
hh0(x) = hh(x, inputs[2], inputs[3])
hh1(y) = hh(inputs[1], y, inputs[3])
hh2(z) = hh(inputs[1], inputs[2], z)

@testset "assemble_forward" begin
    pyout = pycall(fem_eval, PyObject, assemble_fenics, templates, inputs...)
    numpy_output, fenics_output, fenics_inputs, tape = [get(pyout, PyObject, i) for i in 0:3]
    @test pybuiltin(:isinstance)(fenics_output, pyadjoint.AdjFloat)

    u1 = fa.interpolate(fa.Constant(1.0), V)
    J = assemble_fenics(u1, fa.Constant(0.5), fa.Constant(0.6))
    @test np.isclose(numpy_output, J)

    jnumpy_output = PyCall.convert(Float64, numpy_output)
    jJ = PyCall.convert(Float64, J)
    @test isapprox(jnumpy_output, jJ)
end

@testset "assemble_vjp" begin
    pyout = pycall(fem_eval, PyObject, assemble_fenics, templates, inputs...)
    numpy_output, fenics_output, fenics_inputs, tape = [get(pyout, PyObject, i) for i in 0:3]

    g = np.ones_like(numpy_output)
    vjp_out = vjp_fem_eval(g, fenics_output, fenics_inputs, tape)

    fdm = FiniteDifferences.central_fdm(2, 1)

    fdm_jac1 = FiniteDifferences.jacobian(fdm, hh0, inputs[1])
    fdm_jac2 = FiniteDifferences.jacobian(fdm, hh1, inputs[2])
    fdm_jac3 = FiniteDifferences.jacobian(fdm, hh2, inputs[3])

    @test isapprox(vjp_out[1], fdm_jac1[1]', atol=1e-5)
    @test isapprox(vjp_out[2], fdm_jac2[1]', atol=1e-5)
    @test isapprox(vjp_out[3], fdm_jac3[1]', atol=1e-5)
end

mesh = fa.UnitSquareMesh(6, 5)
V = fenics.FunctionSpace(mesh, "P", 1)

function solve_fenics(kappa0, kappa1)

    f = fa.Expression(
        "10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=2
    )

    u = fa.Function(V)
    bcs = [fa.DirichletBC(V, fa.Constant(0.0), "on_boundary")]

    inner, grad, dx = ufl.inner, ufl.grad, ufl.dx
    JJ = 0.5 * inner(kappa0 * grad(u), grad(u)) * dx - kappa1 * f * u * dx
    v = fenics.TestFunction(V)
    F = fenics.derivative(JJ, u, v)
    fa.solve(F == 0, u, bcs=bcs)
    return u
end


templates = (fa.Constant(0.0), fa.Constant(0.0))
inputs = (ones(1) * 0.5, ones(1) * 0.6)

@testset "solve_forward" begin
    numpy_output, _, _, _ = fem_eval(solve_fenics, templates, inputs...)
    u = solve_fenics(fa.Constant(0.5), fa.Constant(0.6))
    @test isapprox(numpy_output, fenics_to_numpy(u))
end

@testset "solve_vjp" begin
    numpy_output, fenics_output, fenics_inputs, tape = fem_eval(
        solve_fenics, templates, inputs...
    )
    # g = np.ones_like(numpy_output)
    g = ones(size(numpy_output))
    vjp_out = vjp_fem_eval(g, fenics_output, fenics_inputs, tape)
    @test isapprox(vjp_out[1], [-2.91792642])
    @test isapprox(vjp_out[2], [2.43160535])
end
