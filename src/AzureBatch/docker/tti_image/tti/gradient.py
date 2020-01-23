# Import modules
from sympy import cos, sin

from devito import clear_cache, Eq, Function, TimeFunction, Inc, Operator
from forward import subsampled
from kernels import kernel_zhang_adj, kernel_zhang_fwd
from sources import Receiver, TimeAxis
from wave_utils import imaging_condition, freesurface, imaging_condition_sg
from staggered_kernels import adjoint_stencil, pressure_fields, src_rec

def gradient(model, save=False, space_order=12, sub=None, fs=False, isic=False):
    clear_cache()

    # Parameters
    s = model.grid.stepping_dim.spacing
    nt = 10
    time_range = TimeAxis(start=0, num=nt, step=1)
    m, damp, epsilon, delta, theta, phi, rho = (model.m, model.damp, model.epsilon,
                                                model.delta, model.theta, model.phi,
                                                model.rho)
    m = m * rho
    # Tilt and azymuth setup
    ang0 = cos(theta)
    ang1 = sin(theta)
    ang2 = cos(phi)
    ang3 = sin(phi)

    # Create the forward wavefield
    f_h = f_t = 1
    if sub is not None and (sub[0] > 1 or sub[1] > 1):
        f_h = sub[1]
        f_t = sub[0]
        u, v = subsampled(model, nt, space_order, t_sub=sub[0], space_sub=sub[1])
    else:
        u = TimeFunction(name='u', grid=model.grid, time_order=2, space_order=space_order,
                         save=nt)
        v = TimeFunction(name='v', grid=model.grid, time_order=2, space_order=space_order,
                         save=nt)
    p = TimeFunction(name='p', grid=model.grid, time_order=2, space_order=space_order)
    q = TimeFunction(name='q', grid=model.grid, time_order=2, space_order=space_order)

    H0, H1 = kernel_zhang_fwd(p, q, ang0, ang1, ang2, ang3, epsilon, delta, rho)

    # Stencils
    s = model.grid.stepping_dim.spacing
    stencilp = damp * (2 * p - damp * p.forward + s**2 / m * H0)
    stencilr = damp * (2 * q - damp * q.forward + s**2 / m * H1)
    first_stencil = Eq(p.backward, stencilp)
    second_stencil = Eq(q.backward, stencilr)
    expression = [first_stencil, second_stencil]

    # Source symbol with input wavelet
    src = Receiver(name='src', grid=model.grid, time_range=time_range, npoint=1)
    src_term = src.inject(field=p.backward, expr=src.dt * s**2 / m)
    src_term += src.inject(field=q.backward, expr=src.dt * s**2 / m)
    expression += src_term

    if fs:
        expression += freesurface(p, model.nbpml, forward=False)
        expression += freesurface(q, model.nbpml, forward=False)
    grad = Function(name="grad", grid=u.grid, space_order=0)
    expression += [Inc(grad, rho * f_t * f_h * imaging_condition(model, u, v, p, q, isic=isic))]

    op = Operator(expression, subs=model.spacing_map, dse='aggressive', dle='advanced',
                  name="gradient")

    return op
