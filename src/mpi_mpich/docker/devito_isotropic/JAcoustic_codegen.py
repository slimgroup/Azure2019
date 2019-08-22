# Acoustic wave equations with Devito
# Forward/adjoint nonlinear and Born modeling
# Authors: Mathias Louboutin, Philipp Witte
# Date: November 2017
#

# Import modules
from __future__ import print_function
import numpy as np
import gc, os, psutil
from numpy.random import randint
from sympy import solve, cos, sin, expand, symbols
from sympy import Function as fint
from devito.logger import set_log_level
from devito import Eq, Function, TimeFunction, Dimension, Operator, clear_cache, ConditionalDimension, DefaultDimension, Inc
from devito import first_derivative, left, right
from PySource import PointSource, Receiver
from PyModel import Model
from checkpoint import DevitoCheckpoint, CheckpointOperator
from pyrevolve import Revolver
from utils import freesurface


def acoustic_laplacian(v, rho):
    if rho is None:
        Lap = v.laplace
        rho = 1
    else:
        if isinstance(rho, Function):
            Lap = sum([first_derivative(first_derivative(v, fd_order=int(v.space_order/2), side=left, dim=d) / rho,
                       fd_order=int(v.space_order/2), dim=d, side=right) for d in v.space_dimensions])
        else:
            Lap = 1 / rho * v.laplace
    return Lap, rho

#def forward_modeling(model, src_coords, wavelet, rec_coords, save=False, space_order=8,
#                     free_surface=False, op_return=False, u_return=False, dt=None, tsub_factor=1):
def forward_modeling(model, geometry, save=False, space_order=8,
                     free_surface=False, op_return=False, u_return=False, dt=None, tsub_factor=1):

    clear_cache()

    # Parameters
    nt = geometry.src.shape[0]
    if dt is None:
        dt = model.critical_dt
    m, rho, damp = model.m, model.rho, model.damp

    # Create the forward wavefield
    if save is False:
        u = TimeFunction(name='u', grid=model.grid, time_order=2, space_order=space_order)
        eqsave = []
    elif save is True and tsub_factor > 1:
        u = TimeFunction(name='u', grid=model.grid, time_order=2, space_order=space_order)
        time_subsampled = ConditionalDimension(name='t_sub', parent=u.grid.time_dim, factor=tsub_factor)
        nsave = (nt-1)//tsub_factor + 2
        usave = TimeFunction(name='us', grid=model.grid, time_order=2, space_order=space_order, time_dim=time_subsampled, save=nsave)
        eqsave = [Eq(usave.forward, u.forward)]
    else:
        u = TimeFunction(name='u', grid=model.grid, time_order=2, space_order=space_order, save=nt)
        eqsave = []

    # Set up PDE and rearrange
    ulaplace, rho = acoustic_laplacian(u, rho)
    stencil = damp * ( 2.0 * u - damp * u.backward + dt**2 * rho / m * ulaplace)

    # Rearrange expression
    expression = [Eq(u.forward, stencil)]

    # Data is sampled at receiver locations
    rec_modeled = geometry.rec
    rec_term = rec_modeled.interpolate(expr=u)
    expression += rec_term

    # Create operator and run
    if save:
        expression += eqsave

    # Free surface
    kwargs = dict()
    if free_surface is True:
        expression += freesurface(u, space_order//2, model.nbpml)

    # Source symbol with input wavelet
    src_term = geometry.src.inject(field=u.forward, expr=geometry.src.dt * rho * dt**2 / m)
    expression += src_term

    subs = model.spacing_map
    subs[u.grid.time_dim.spacing] = dt
    op = Operator(expression, subs=subs, dse='advanced', dle='advanced')
    if op_return is False:
        cf = op.cfunction
        summary = op.apply()
        if save is True and tsub_factor > 1:
            return rec_modeled, usave#, summary
        else:
            return rec_modeled, u#, summary
    else:
        if u_return is False:
            return op
        else:
            if tsub_factor > 1:
                return op, usave
            else:
                return op, u


# def adjoint_modeling(model, src_coords, rec_coords, rec_data, space_order=8, free_surface=False, dt=None):
def adjoint_modeling(model, geometry, space_order=8, free_surface=False, dt=None):

    clear_cache()

    # Parameters
    nt = geometry.rec.shape[0]
    if dt is None:
        dt = model.critical_dt
    m, rho, damp = model.m, model.rho, model.damp

    # Create the adjoint wavefield
    v = TimeFunction(name="v", grid=model.grid, time_order=2, space_order=space_order)

    # Set up PDE and rearrange
    vlaplace, rho = acoustic_laplacian(v, rho)

    stencil = damp * (2.0 * v - damp * v.forward + dt**2 * rho / m * vlaplace)
    expression = [Eq(v.backward, stencil)]

    # Free surface
    if free_surface is True:
        expression += freesurface(v, space_order//2, model.nbpml, forward=False)

    # Adjoint source is injected at receiver locations
    adj_src = geometry.rec.inject(field=v.backward, expr=geometry.rec * rho * dt**2 / m)
    expression += adj_src

    # Data is sampled at source locations
    src_modeled = geometry.src
    adj_rec = src_modeled.interpolate(expr=v)
    expression += adj_rec

    subs = model.spacing_map
    subs[v.grid.time_dim.spacing] = dt
    op = Operator(expression, subs=subs, dse='advanced', dle='advanced')
    cf = op.cfunction
    summary = op.apply()

    return src_modeled#, summary


#def forward_born(model, src_coords, wavelet, rec_coords, space_order=8, isic=False, dt=None, free_surface=False):
def forward_born(model, geometry, space_order=8, isic=False, dt=None, free_surface=False):

    clear_cache()

    # Parameters
    nt = geometry.src.shape[0]
    if dt is None:
        dt = model.critical_dt
    m, rho, dm, damp = model.m, model.rho, model.dm, model.damp

    # Create the forward and linearized wavefield
    u = TimeFunction(name="u", grid=model.grid, time_order=2, space_order=space_order)
    du = TimeFunction(name="du", grid=model.grid, time_order=2, space_order=space_order)
    if len(model.shape) == 2:
        x,y = u.space_dimensions
    else:
        x,y,z = u.space_dimensions

    # Set up PDEs and rearrange
    ulaplace, rho = acoustic_laplacian(u, rho)
    dulaplace, _ = acoustic_laplacian(du, rho)

    if isic:
        # Sum ((u.dx * d, / rho).dx for x in dimensions)
        # space_order//2  so that u.dx.dx has the same radius as u.laplace
        du_aux = sum([first_derivative(first_derivative(u, dim=d, fd_order=space_order//2) * dm / rho,
                                       fd_order=space_order//2, dim=d)
                      for d in u.space_dimensions])
        lin_source = dm /rho * u.dt2 * m - du_aux
    else:
        lin_source = dm / rho * u.dt2

    stencil_u = damp * (2.0 * u - damp * u.backward + dt**2 * rho / m * ulaplace)
    stencil_du = damp * (2.0 * du - damp * du.backward + dt**2 * rho / m * (dulaplace - lin_source))

    expression_u = [Eq(u.forward, stencil_u)]
    expression_du = [Eq(du.forward, stencil_du)]
    expression = expression_u + expression_du

    # Define source symbol with wavelet
    src_term = geometry.src.inject(field=u.forward, expr=geometry.src * rho * dt**2 / m)
    expression += src_term

    # Define receiver symbol
    rec_modeled = geometry.rec
    rec_term = rec_modeled.interpolate(expr=du)
    expression += rec_term

    # Free surface
    if free_surface is True:
        expression += freesurface(u, space_order//2, model.nbpml)
        expression += freesurface(du, space_order//2, model.nbpml)

    subs = model.spacing_map
    subs[u.grid.time_dim.spacing] = dt
    op = Operator(expression, subs=subs, dse='advanced', dle='advanced')
    cf = op.cfunction
    summary = op.apply()

    return rec_modeled# summary


#def adjoint_born(model, rec_coords, rec_data, u=None, op_forward=None, is_residual=False,
#                 space_order=8, isic=False, dt=None, n_checkpoints=None, maxmem=None,
#                 free_surface=False, tsub_factor=1, checkpointing=False):
def adjoint_born(model, rec_g, u=None, op_forward=None, space_order=8, isic=False,
                 dt=None, n_checkpoints=None, maxmem=None,free_surface=False, tsub_factor=1,
                 checkpointing=False):

    clear_cache()

    # Parameters
    nt = rec_g.shape[0]
    if dt is None:
        dt = model.critical_dt
    m, rho, damp = model.m, model.rho, model.damp

    # Create adjoint wavefield and gradient
    v = TimeFunction(name='v', grid=model.grid, time_order=2, space_order=space_order)
    gradient = Function(name='gradient', grid=model.grid)

    # Set up PDE and rearrange
    vlaplace, rho = acoustic_laplacian(v, rho)
    stencil = damp * (2.0 * v - damp * v.forward + dt**2 * rho / m * vlaplace)
    expression = [Eq(v.backward, stencil)]

    # Data at receiver locations as adjoint source
    adj_src = rec_g.inject(field=v.backward, expr=rec_g * rho * dt**2 / m)

    # Gradient update
    if u is None:
        u = TimeFunction(name='u', grid=model.grid, time_order=2, space_order=space_order)
    if isic is not True:
        gradient_update = [Inc(gradient, - dt * u.dt2 / rho * v)]
    else:
        # sum u.dx * v.dx fo x in dimensions.
        # space_order//2
        diff_u_v = sum([first_derivative(u, dim=d, fd_order=space_order//2)*
                        first_derivative(v, dim=d, fd_order=space_order//2)
                        for d in u.space_dimensions])
        gradient_update = [Inc(gradient, - tsub_factor * dt * (u * v.dt2 * m + diff_u_v) / rho)]

    # Free surface
    if free_surface is True:
        expression += freesurface(v, space_order//2, model.nbpml, forward=False)

    expression += gradient_update + adj_src
    subs = model.spacing_map
    subs[u.grid.time_dim.spacing] = dt

    if isic is True:
        op = Operator(expression, subs=subs, dse='noop', dle='noop')
    else:
        op = Operator(expression, subs=subs, dse='advanced', dle='advanced')

    # Optimal checkpointing
    summary1 = None
    summary2 = None
    if op_forward is not None and checkpointing is False:

        # Compile first
        cf1 = op_forward.cfunction
        cf2 = op.cfunction

        # Run forward and adjoint
        summary1 = op_forward.apply()
        summary2 = op.apply()
    else:
        cf = op.cfunction
        summary1 = op.apply()
    clear_cache()

    return gradient

########################################################################################################################

# def forward_freq_modeling(model, src_coords, wavelet, rec_coords, freq, space_order=8, dt=None, factor=None, free_surface=False):
#     # Forward modeling with on-the-fly DFT of forward wavefields
#     clear_cache()
#
#     # Parameters
#     nt = wavelet.shape[0]
#     if dt is None:
#         dt = model.critical_dt
#     m, rho, damp = model.m, model.rho, model.damp
#
#     freq_dim = Dimension(name='freq_dim')
#     time = model.grid.time_dim
#     if factor is None:
#         factor = int(1 / (dt*4*np.max(freq)))
#         tsave = ConditionalDimension(name='tsave', parent=model.grid.time_dim, factor=factor)
#     if factor==1:
#         tsave = time
#     else:
#         tsave = ConditionalDimension(name='tsave', parent=model.grid.time_dim, factor=factor)
#     print("DFT subsampling factor: ", factor)
#
#     # Create wavefields
#     nfreq = freq.shape[0]
#     u = TimeFunction(name='u', grid=model.grid, time_order=2, space_order=space_order)
#     f = Function(name='f', dimensions=(freq_dim,), shape=(nfreq,))
#     f.data[:] = freq[:]
#     ufr = Function(name='ufr', dimensions=(freq_dim,) + u.indices[1:], shape=(nfreq,) + model.shape_domain)
#     ufi = Function(name='ufi', dimensions=(freq_dim,) + u.indices[1:], shape=(nfreq,) + model.shape_domain)
#
#     ulaplace, rho = acoustic_laplacian(u, rho)
#
#     # Set up PDE and rearrange
#     stencil = damp * (2.0 * u - damp * u.backward + dt**2 * rho / m * ulaplace)
#     expression = [Eq(u.forward, stencil)]
#     expression += [Eq(ufr, ufr + factor*u*cos(2*np.pi*f*tsave*factor*dt))]
#     expression += [Eq(ufi, ufi - factor*u*sin(2*np.pi*f*tsave*factor*dt))]
#
#     # Source symbol with input wavelet
#     src = PointSource(name='src', grid=model.grid, ntime=nt, coordinates=src_coords)
#     src.data[:] = wavelet[:]
#     src_term = src.inject(field=u.forward, expr=src * dt**2 / m)
#
#     # Data is sampled at receiver locations
#     rec = Receiver(name='rec', grid=model.grid, ntime=nt, coordinates=rec_coords)
#     rec_term = rec.interpolate(expr=u)
#
#     # Create operator and run
#
#     expression += src_term + rec_term
#     # Free surface
#     if free_surface is True:
#         expression += freesurface(u, space_order//2, model.nbpml)
#
#     subs = model.spacing_map
#     subs[u.grid.time_dim.spacing] = dt
#     op = Operator(expression, subs=subs, dse='advanced', dle='advanced')
#     cf = op.cfunction
#     op()
#
#     return rec.data, ufr, ufi
#
#
# def adjoint_freq_born(model, rec_coords, rec_data, freq, ufr, ufi, space_order=8, dt=None, isic=False, factor=None,
#                       free_surface=False):
#     clear_cache()
#
#     # Parameters
#     nt = rec_data.shape[0]
#     if dt is None:
#         dt = model.critical_dt
#     m, rho, damp = model.m, model.rho, model.damp
#     nfreq = ufr.shape[0]
#     time = model.grid.time_dim
#     if factor is None:
#         factor = int(1 / (dt*4*np.max(freq)))
#         tsave = ConditionalDimension(name='tsave', parent=model.grid.time_dim, factor=factor)
#     if factor==1:
#         tsave = time
#     else:
#         tsave = ConditionalDimension(name='tsave', parent=model.grid.time_dim, factor=factor)
#     dtf = factor * dt
#     ntf = factor / nt
#     print("DFT subsampling factor: ", factor)
#
#     # Create the forward and adjoint wavefield
#     v = TimeFunction(name='v', grid=model.grid, time_order=2, space_order=space_order)
#     f = Function(name='f', dimensions=(ufr.indices[0],), shape=(nfreq,))
#     f.data[:] = freq[:]
#     gradient = Function(name="gradient", grid=model.grid)
#     vlaplace, rho = acoustic_laplacian(v, rho)
#
#     # Set up PDE and rearrange
#     stencil = damp * (2.0 * v - damp * v.forward + dt**2 * rho / m * vlaplace)
#     expression = [Eq(v.backward, stencil)]
#
#     # Data at receiver locations as adjoint source
#     rec = Receiver(name='rec', grid=model.grid, ntime=nt, coordinates=rec_coords)
#     rec.data[:] = rec_data[:]
#     adj_src = rec.inject(field=v.backward, expr=rec * dt**2 / m)
#
#     # Gradient update
#     if isic is True:
#         if len(model.shape) == 2:
#             gradient_update = [Eq(gradient, gradient + (2*np.pi*f)**2*ntf*(ufr*cos(2*np.pi*f*tsave*dtf) - ufi*sin(2*np.pi*f*tsave*dtf))*v*model.m -
#                                                        (ufr.dx*cos(2*np.pi*f*tsave*dtf) - ufi.dx*sin(2*np.pi*f*tsave*dtf))*v.dx*ntf -
#                                                        (ufr.dy*cos(2*np.pi*f*tsave*dtf) - ufi.dy*sin(2*np.pi*f*tsave*dtf))*v.dy*ntf)]
#         else:
#             gradient_update = [Eq(gradient, gradient + (2*np.pi*f)**2*ntf*(ufr*cos(2*np.pi*f*tsave*dtf) - ufi*sin(2*np.pi*f*tsave*dtf))*v*model.m -
#                                                        (ufr.dx*cos(2*np.pi*f*tsave*dtf) - ufi.dx*sin(2*np.pi*f*tsave*dtf))*v.dx*ntf -
#                                                        (ufr.dy*cos(2*np.pi*f*tsave*dtf) - ufi.dy*sin(2*np.pi*f*tsave*dtf))*v.dy*ntf -
#                                                        (ufr.dz*cos(2*np.pi*f*tsave*dtf) - ufi.dz*sin(2*np.pi*f*tsave*dtf))*v.dz*ntf)]
#     else:
#         gradient_update = [Eq(gradient, gradient + (2*np.pi*f)**2/nt*(ufr*cos(2*np.pi*f*tsave*dtf) - ufi*sin(2*np.pi*f*tsave*dtf))*v)]
#
#     # Create operator and run
#
#     # Free surface
#     if free_surface is True:
#         expression += freesurface(v, space_order//2, model.nbpml, forward=False)
#     expression += adj_src + gradient_update
#     subs = model.spacing_map
#     subs[v.grid.time_dim.spacing] = dt
#     op = Operator(expression, subs=subs, dse='advanced', dle='advanced')
#     cf = op.cfunction
#     op()
#     clear_cache()
#     return gradient.data
