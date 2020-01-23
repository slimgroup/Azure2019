# Import modules
from sympy import cos, sin

from devito import clear_cache, Grid, Eq, TimeFunction, Operator, ConditionalDimension, Function
from kernels import kernel_zhang_fwd
from sources import Receiver, TimeAxis
from staggered_kernels import *
from wave_utils import freesurface, linearized_source, linearized_source_sg, subsampled

def born(model, save=False, space_order=12, sub=None, recu=False, fs=False, isic=False):
    clear_cache()

    # Parameters
    s = model.grid.stepping_dim.spacing
    nt = 10
    time_range = TimeAxis(start=0, num=nt, step=1)
    m, damp, epsilon, delta, theta, phi, rho = (model.m, model.damp, model.epsilon,
                                                model.delta, model.theta, model.phi,
                                                model.rho)
    m = rho * m
    # Tilt and azymuth setup
    ang0 = cos(theta)
    ang1 = sin(theta)
    ang2 = cos(phi)
    ang3 = sin(phi)

    # Create the forward wavefield
    if sub is not None and (sub[0] > 1 or sub[1] > 1):
        usave, vsave = subsampled(model, nt, space_order, t_sub=sub[0], space_sub=sub[1])
        u = TimeFunction(name='u', grid=model.grid, time_order=2, space_order=space_order)
        v = TimeFunction(name='v', grid=model.grid, time_order=2, space_order=space_order)
        eq_save = [Eq(usave, u), Eq(vsave, v)]
    elif save:
        u = TimeFunction(name='u', grid=model.grid, time_order=2, space_order=space_order,
                         save=nt)
        v = TimeFunction(name='v', grid=model.grid, time_order=2, space_order=space_order,
                         save=nt)
        eq_save = []
    else:
        u = TimeFunction(name='u', grid=model.grid, time_order=2, space_order=space_order)
        v = TimeFunction(name='v', grid=model.grid, time_order=2, space_order=space_order)
        eq_save = []

    # Create the linearized  wavefield'
    ul = TimeFunction(name='ul', grid=model.grid, time_order=2, space_order=space_order)
    vl = TimeFunction(name='vl', grid=model.grid, time_order=2, space_order=space_order)

    # FD kernels
    H0, H1 = kernel_zhang_fwd(u, v, ang0, ang1, ang2, ang3, epsilon, delta, rho)
    H0l, H1l = kernel_zhang_fwd(ul, vl, ang0, ang1, ang2, ang3, epsilon, delta, rho)
    # Stencils
    s = model.grid.stepping_dim.spacing
    # wavefield
    stencilp = damp * (2 * u - damp * u.backward + s**2 / m * H0)
    stencilr = damp * (2 * v - damp * v.backward + s**2 / m * H1)
    first_stencil = Eq(u.forward, stencilp)
    second_stencil = Eq(v.forward, stencilr)
    expression = [first_stencil, second_stencil]

    # Source symbol with input wavelet
    src = Receiver(name='src', grid=model.grid, time_range=time_range, npoint=1)
    src_term = src.inject(field=u.forward, expr=src.dt * s**2 / m)
    src_term += src.inject(field=v.forward, expr=src.dt * s**2 / m)
    expression += src_term

    # Linearized wavefield
    lin_q = linearized_source(model, u, v, isic=isic)
    stencilp = damp * (2 * ul - damp * ul.backward + s**2 / m * (H0l + lin_q[0]))
    stencilr = damp * (2 * vl - damp * vl.backward + s**2 / m * (H1l + lin_q[1]))
    first_stencil = Eq(ul.forward, stencilp)
    second_stencil = Eq(vl.forward, stencilr)
    expression += [first_stencil, second_stencil]

    if fs:
        expression += freesurface(u, model.nbpml)
        expression += freesurface(v, model.nbpml)


    rec = Receiver(name='rec', grid=model.grid, time_range=time_range, npoint=2)
    expression += rec.interpolate(expr=ul + vl)
    if recu:
        recu = Receiver(name='recu', grid=model.grid, time_range=time_range, npoint=2)
        expression += recu.interpolate(expr=u + v)
    kwargs = {'dse': 'aggressive', 'dle': 'advanced'}
    op = Operator(expression + eq_save, subs=model.spacing_map,
                  name="born", **kwargs)

    return op
