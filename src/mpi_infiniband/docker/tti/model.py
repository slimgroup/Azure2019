import numpy as np

from sympy import sin, Abs
from devito import (Grid, Inc, Operator, Function, SubDomain, Eq, SubDimension,
                    ConditionalDimension, switchconfig)
from devito.tools import memoized_meth


__all__ = ['Model']


class PhysicalDomain(SubDomain):

    name = 'phydomain'

    def __init__(self, nbpml):
        super(PhysicalDomain, self).__init__()
        self.nbpml = nbpml

    def define(self, dimensions):
        return {d: ('middle', self.nbpml, self.nbpml) for d in dimensions}


# @switchconfig(log_level='ERROR')
def initialize_damp(damp, nbpml, spacing, mask=False):
    """
    Initialise damping field with an absorbing PML layer.
    Parameters
    ----------
    damp : Function
        The damping field for absorbing boundary condition.
    nbpml : int
        Number of points in the damping layer.
    spacing :
        Grid spacing coefficient.
    mask : bool, optional
        whether the dampening is a mask or layer.
        mask => 1 inside the domain and decreases in the layer
        not mask => 0 inside the domain and increase in the layer
    """
    dampcoeff = 1.5 * np.log(1.0 / 0.001) / (40)

    eqs = [Eq(damp, 1.0)] if mask else []
    for d in damp.dimensions:
        # left
        dim_l = SubDimension.left(name='abc_%s_l' % d.name, parent=d,
                                  thickness=nbpml)
        pos = Abs((nbpml - (dim_l - d.symbolic_min) + 1) / float(nbpml))
        val = dampcoeff * (pos - sin(2*np.pi*pos)/(2*np.pi))
        val = -val if mask else val
        eqs += [Inc(damp.subs({d: dim_l}), val/d.spacing)]
        # right
        dim_r = SubDimension.right(name='abc_%s_r' % d.name, parent=d,
                                   thickness=nbpml)
        pos = Abs((nbpml - (d.symbolic_max - dim_r) + 1) / float(nbpml))
        val = dampcoeff * (pos - sin(2*np.pi*pos)/(2*np.pi))
        val = -val if mask else val
        eqs += [Inc(damp.subs({d: dim_r}), val/d.spacing)]

    Operator(eqs, name='initdamp')()


def initialize_function(function, data, nbpml, pad_mode='edge'):
    """
    Initialize a `Function` with the given ``data``. ``data``
    does *not* include the PML layers for the absorbing boundary conditions;
    these are added via padding by this function.
    Parameters
    ----------
    function : Function
        The initialised object.
    data : ndarray
        The data array used for initialisation.
    nbpml : int
        Number of PML layers for boundary damping.
    pad_mode : str or callable, optional
        A string or a suitable padding function as explained in :func:`numpy.pad`.
    """
    pad_widths = [(nbpml + i.left, nbpml + i.right) for i in function._size_halo]
    data = np.pad(data, pad_widths, pad_mode)
    function.data_with_halo[:] = data


class Model(object):
    """The physical model used in seismic inversion processes.
    :param origin: Origin of the model in m as a tuple in (x,y,z) order
    :param spacing: Grid size in m as a Tuple in (x,y,z) order
    :param shape: Number of grid points size in (x,y,z) order
    :param vp: Velocity in km/s
    :param nbpml: The number of PML layers for boundary damping
    :param dm: Model perturbation in s^2/km^2
    The :class:`Model` provides two symbolic data objects for the
    creation of seismic wave propagation operators:
    :param m: The square slowness of the wave
    :param damp: The damping field for absorbing boundarycondition
    """
    def __init__(self, origin, spacing, shape, vp, rho=1, nbpml=40, dtype=np.float32,
                 dm=None, epsilon=None, delta=None, theta=None, phi=None,
                 space_order=8, in_dim=None, dt=None):

        self._dt = dt
        self.shape = shape
        self.nbpml = int(nbpml)
        self.origin = tuple([dtype(o) for o in origin])
        self._is_tti = False
        # Origin of the computational domain with PML to inject/interpolate
        # at the correct index
        origin_pml = tuple([dtype(o - s*nbpml) for o, s in zip(origin, spacing)])
        phydomain = PhysicalDomain(self.nbpml)
        shape_pml = np.array(shape) + 2 * self.nbpml
        # Physical extent is calculated per cell, so shape - 1
        extent = tuple(np.array(spacing) * (shape_pml - 1))
        self.grid = Grid(extent=extent, shape=shape_pml, origin=origin_pml, dtype=dtype,
                         dimensions=in_dim)

        # Create square slowness of the wave as symbol `m`
        if isinstance(vp, np.ndarray):
            self.m = Function(name="m", grid=self.grid, space_order=space_order)
        else:
            self.m = 1/vp**2
        self._physical_parameters = ('m',)

        if isinstance(rho, np.ndarray):
            self._physical_parameters += ('rho',)
            self.rho = Function(name="rho", grid=self.grid, space_order=space_order)
            initialize_function(self.rho, rho, self.nbpml)
        else:
            self.rho = rho

        # Set model velocity, which will also set `m`
        self.vp = vp

        # Create dampening field as symbol `damp`
        self.damp = Function(name="damp", grid=self.grid)
        initialize_damp(self.damp, self.nbpml, self.spacing, mask=True)
        self._physical_parameters += ('damp',)

        # Additional parameter fields for TTI operators
        self.scale = 1.

        if dm is not None:
            self.dm = Function(name="dm", grid=self.grid, space_order=space_order)
            initialize_function(self.dm, dm, self.nbpml)
        else:
            self.dm = 1

        if epsilon is not None:
            self._is_tti = True
            if isinstance(epsilon, np.ndarray):
                self._physical_parameters += ('epsilon',)
                self.epsilon = Function(name="epsilon", grid=self.grid,
                                        space_order=space_order)
                initialize_function(self.epsilon, epsilon, self.nbpml)
                # Maximum velocity is scale*max(vp) if epsilon > 0
                if np.max(self.epsilon.data) > 0:
                    self.scale = np.sqrt(np.max(1 + 2 * self.epsilon.data))
            else:
                self.epsilon = epsilon
                self.scale = np.sqrt(1 + 2 * self.epsilon)
        else:
            self.epsilon = 0.0
            self.scale = 1.0

        if delta is not None:
            self._is_tti = True
            if isinstance(delta, np.ndarray):
                self._physical_parameters += ('delta',)
                self.delta = Function(name="delta", grid=self.grid,
                                      space_order=space_order)
                initialize_function(self.delta, delta, self.nbpml)
            else:
                self.delta = delta
        else:
            self.delta = 0.0

        if theta is not None:
            self._is_tti = True
            if isinstance(theta, np.ndarray):
                self._physical_parameters += ('theta',)
                self.theta = Function(name="theta", grid=self.grid,
                                      space_order=space_order)
                initialize_function(self.theta, theta, self.nbpml)
            else:
                self.theta = theta
        else:
            self.theta = 0.0

        if phi is not None:
            self._is_tti = True
            if isinstance(phi, np.ndarray):
                self._physical_parameters += ('phi',)
                self.phi = Function(name="phi", grid=self.grid,
                                    space_order=space_order)
                initialize_function(self.phi, phi, self.nbpml)
            else:
                self.phi = phi
        else:
            self.phi = 0.0

    @property
    def is_tti(self):
        return self._is_tti

    @property
    def dim(self):
        """
        Spatial dimension of the problem and model domain.
        """
        return self.grid.dim

    @property
    def spacing(self):
        """
        Grid spacing for all fields in the physical model.
        """
        return self.grid.spacing

    @property
    def spacing_map(self):
        """
        Map between spacing symbols and their values for each :class:`SpaceDimension`
        """
        subs = self.grid.spacing_map
        subs[self.grid.time_dim.spacing] = self.critical_dt
        return subs

    @property
    def dtype(self):
        """
        Data type for all assocaited data objects.
        """
        return self.grid.dtype

    @property
    def shape_domain(self):
        """Computational shape of the model domain, with PML layers"""
        return self.grid.shape

    @property
    def domain_size(self):
        """
        Physical size of the domain as determined by shape and spacing
        """
        return tuple((d-1) * s for d, s in zip(self.shape, self.spacing))

    @property
    def critical_dt(self):
        """Critical computational time step value from the CFL condition."""
        # For a fixed time order this number goes down as the space order increases.
        #
        # The CFL condtion is then given by
        # dt <= coeff * h / (max(velocity))
        coeff = 0.38 if len(self.shape) == 3 else 0.42
        dt = self.dtype(coeff * np.min(self.spacing) / (self.scale*np.max(self.vp)))
        return self._dt or self.dtype('%.3f' % dt)

    @property
    def vp(self):
        """:class:`numpy.ndarray` holding the model velocity in km/s.
        .. note::
        Updating the velocity field also updates the square slowness
        ``self.m``. However, only ``self.m`` should be used in seismic
        operators, since it is of type :class:`Function`.
        """
        return self._vp

    @vp.setter
    def vp(self, vp):
        """Set a new velocity model and update square slowness
        :param vp : new velocity in km/s
        """
        self._vp = vp

        # Update the square slowness according to new value
        if isinstance(vp, np.ndarray):
            initialize_function(self.m, 1 / (self.vp * self.vp), self.nbpml)
        else:
            self.m.data = 1 / vp**2

    def physical_params(self, **kwargs):
        """
        Return all set physical parameters and update to input values if provided
        """
        known = [getattr(self, i) for i in self._physical_parameters]

        params = {i.name: kwargs.get(i.name, i) or i for i in known}
        return params

    @memoized_meth
    def subgrid(self, factor=1):
        sub_dim = []
        for dim in self.grid.dimensions:
            sub_dim += [ConditionalDimension(dim.name + 'sub', parent=dim, factor=factor)]
        grid = Grid(shape=tuple([i//factor for i in self.shape]),
                    comm=self.grid.distributor.comm,
                    extent=self.grid.extent,
                    dimensions=tuple(sub_dim))
        return grid
