import numpy as np

from devito import (SubDimension, Eq, grad, first_derivative,
                    Function, ConditionalDimension, TimeFunction)

__all__ = ['freesurface']

def freesurface(field, npml, forward=True):
    """
    Generate the stencil that mirrors the field as a free surface modeling for
    the acoustic wave equation
    """
    d = field.grid.dimensions[-1]
    fs = SubDimension.left(name='fs', parent=d, thickness=npml)

    field_m = field.forward if forward else field.backward
    lhs = field_m.subs({d: fs})
    rhs = -field_m.subs({d: 2*npml - fs})
    return [Eq(lhs, rhs)]


def linearized_source(model, u, v, isic=False):
    """
    Linearized data source
    """
    dm = model.dm
    rho = model.rho
    m = model.m
    if isic:
        return (dm * rho * u.dt2 * m - div(dm * rho * grad(u)),
                dm * rho * v.dt2 * m - div(dm * rho * grad(v)))
    else:
        return -rho * dm * u.dt2, -rho * dm * v.dt2


def imaging_condition(model, u, v, p, q, isic=False):
    """
    Graident or imaging condition
    """
    s = model.grid.time_dim.spacing
    if isic:
        m = model.m
        grads = grad(u).T*grad(p) + grad(v).T*grad(q)
        return - s * ((u * p.dt2 + v * q.dt2) * m + grads)
    else:
        return - s * (u * p.dt2 + v * q.dt2)


def subsampled(model, nt, space_order, t_sub=2, space_sub=2):
    if t_sub == 1 and space_sub == 1:
        return

    if t_sub > 1:
        time_subsampled = ConditionalDimension('t_sub', parent=model.grid.time_dim,
                                               factor=t_sub)
        nsave = (nt-1)//t_sub + 2
    else:
        time_subsampled = model.grid.time_dim
        nsave = nt

    if space_sub > 1:
        grid2 = model.subgrid(factor=space_sub)
    else:
        grid2 = model.grid

    usave = TimeFunction(name='us', grid=grid2, time_order=2, space_order=space_order,
                         time_dim=time_subsampled, save=nsave)
    vsave = TimeFunction(name='vs', grid=grid2, time_order=2, space_order=space_order,
                         time_dim=time_subsampled, save=nsave)

    return usave, vsave