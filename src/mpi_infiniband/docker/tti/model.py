import numpy as np

from sympy import sin, Abs
from devito import (Grid, Inc, Operator, Function, SubDomain, Eq, SubDimension,
                    ConditionalDimension, switchconfig)
from devito.tools import memoized_meth


__all__ = ['Model']


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

    # TODO: Figure out why yask doesn't like it with dse/dle
    Operator(eqs, name='initdamp', dse='noop', dle='noop')()


def initialize_function(function, data, nbpml):
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
    """
    slices = tuple([slice(nbpml, -nbpml) for _ in range(function.grid.dim)])
    function.data[slices] = data
    eqs = []

    for d in function.dimensions:
        dim_l = SubDimension.left(name='abc_%s_l' % d.name, parent=d,
                                  thickness=nbpml)
        to_copy = nbpml
        eqs += [Eq(function.subs({d: dim_l}), function.subs({d: to_copy}))]
        dim_r = SubDimension.right(name='abc_%s_r' % d.name, parent=d,
                                   thickness=nbpml)
        to_copy = d.symbolic_max - nbpml
        eqs += [Eq(function.subs({d: dim_r}), function.subs({d: to_copy}))]

    # TODO: Figure out why yask doesn't like it with dse/dle
    Operator(eqs, name='padfunc', dse='noop', dle='noop')()


class PhysicalDomain(SubDomain):

    name = 'phydomain'

    def __init__(self, nbpml):
        super(PhysicalDomain, self).__init__()
        self.nbpml = nbpml

    def define(self, dimensions):
        return {d: ('middle', self.nbpml, self.nbpml) for d in dimensions}


class GenericModel(object):
    """
    General model class with common properties
    """
    def __init__(self, origin, spacing, shape, space_order, nbpml=20,
                 dtype=np.float32, subdomains=(), damp_mask=True):
        self.shape = shape
        self.nbpml = int(nbpml)
        self.origin = tuple([dtype(o) for o in origin])

        # Origin of the computational domain with PML to inject/interpolate
        # at the correct index
        origin_pml = tuple([dtype(o - s*nbpml) for o, s in zip(origin, spacing)])
        phydomain = PhysicalDomain(self.nbpml)
        subdomains = subdomains + (phydomain, )
        shape_pml = np.array(shape) + 2 * self.nbpml
        # Physical extent is calculated per cell, so shape - 1
        extent = tuple(np.array(spacing) * (shape_pml - 1))
        self.grid = Grid(extent=extent, shape=shape_pml, origin=origin_pml, dtype=dtype,
                         subdomains=subdomains)

        # Create dampening field as symbol `damp`
        self.damp = Function(name="damp", grid=self.grid)
        initialize_damp(self.damp, self.nbpml, self.spacing, mask=damp_mask)
        self._physical_parameters = ['damp']

    def physical_params(self, **kwargs):
        """
        Return all set physical parameters and update to input values if provided
        """
        is_born = kwargs.pop('is_born', False)
        known = [getattr(self, i) for i in self.physical_parameters]
        dict = {i.name: kwargs.get(i.name, i) or i for i in known}
        if not is_born and 'dm' in dict.keys():
            dict.pop('dm')
        return dict

    def _gen_phys_param(self, field, name, space_order, is_param=False,
                        default_value=0, init_empty=False):
        if field is None and not init_empty:
            return default_value
        if isinstance(field, np.ndarray) or init_empty:
            function = Function(name=name, grid=self.grid, space_order=space_order,
                                parameter=is_param)
            if not init_empty:
                if name is 'rho':
                    initialize_function(function, 1/field, self.nbpml)
                else:
                    initialize_function(function, field, self.nbpml)            
        else:
            function = Constant(name=name, value=field)
        self._physical_parameters.append(name)
        return function

    @property
    def physical_parameters(self):
        return tuple(self._physical_parameters)

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
    def space_dimensions(self):
        """
        Spatial dimensions of the grid
        """
        return self.grid.dimensions

    @property
    def spacing_map(self):
        """
        Map between spacing symbols and their values for each `SpaceDimension`.
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
    def domain_size(self):
        """
        Physical size of the domain as determined by shape and spacing
        """
        return tuple((d-1) * s for d, s in zip(self.shape, self.spacing))


class Model(GenericModel):
    """
    The physical model used in seismic inversion processes.

    Parameters
    ----------
    origin : tuple of floats
        Origin of the model in m as a tuple in (x,y,z) order.
    spacing : tuple of floats
        Grid size in m as a Tuple in (x,y,z) order.
    shape : tuple of int
        Number of grid points size in (x,y,z) order.
    space_order : int
        Order of the spatial stencil discretisation.
    vp : array_like or float
        Velocity in km/s.
    nbpml : int, optional
        The number of PML layers for boundary damping.
    dtype : np.float32 or np.float64
        Defaults to 32.
    epsilon : array_like or float, optional
        Thomsen epsilon parameter (0<epsilon<1).
    delta : array_like or float
        Thomsen delta parameter (0<delta<1), delta<epsilon.
    theta : array_like or float
        Tilt angle in radian.
    phi : array_like or float
        Asymuth angle in radian.

    The `Model` provides two symbolic data objects for the
    creation of seismic wave propagation operators:

    m : array_like or float
        The square slowness of the wave.
    damp : Function
        The damping field for absorbing boundary condition.
    """
    def __init__(self, origin, spacing, shape, space_order, vp=None, nbpml=20,
                 dtype=np.float32, epsilon=None, delta=None, theta=None, phi=None,
                 subdomains=(), dm=None, rho=None, **kwargs):
        super(Model, self).__init__(origin, spacing, shape, space_order, nbpml, dtype,
                                    subdomains)
        tti_empty = kwargs.get('init_tti', False)
        self._dt = kwargs.get('dt', None)
        # Create square slowness of the wave as symbol `m`
        self._vp = self._gen_phys_param(vp, 'vp', space_order, init_empty=tti_empty)
        self.rho = self._gen_phys_param(rho, 'rho', space_order, init_empty=tti_empty, default_value=1.0)
        self._max_vp = kwargs.get('max_vp', np.max(vp))

        # Additional parameter fields for TTI operators
        self.epsilon = self._gen_phys_param(epsilon, 'epsilon', space_order, init_empty=tti_empty)
        self.scale = kwargs.get('scale', 1 if epsilon is None else np.sqrt(1 + 2 * np.max(epsilon)))

        self.delta = self._gen_phys_param(delta, 'delta', space_order, init_empty=tti_empty)
        self.theta = self._gen_phys_param(theta, 'theta', space_order, init_empty=tti_empty)
        self.phi = self._gen_phys_param(phi, 'phi', space_order, init_empty=tti_empty)

        self.dm = self._gen_phys_param(dm, 'dm', space_order, init_empty=tti_empty)

    @property
    def critical_dt(self):
        """
        Critical computational time step value from the CFL condition.
        """
        # For a fixed time order this number decreases as the space order increases.
        #
        # The CFL condtion is then given by
        # dt <= coeff * h / (max(velocity))
        coeff = 0.38 if len(self.shape) == 3 else 0.42
        dt = self.dtype(coeff * np.min(self.spacing) / (self.scale*self._max_vp))
        return self.dtype("%.3f" % dt)

    @property
    def vp(self):
        """
        `numpy.ndarray` holding the model velocity in km/s.

        Notes
        -----
        Updating the velocity field also updates the square slowness
        ``self.m``. However, only ``self.m`` should be used in seismic
        operators, since it is of type `Function`.
        """
        return self._vp

    @vp.setter
    def vp(self, vp):
        """
        Set a new velocity model and update square slowness.

        Parameters
        ----------
        vp : float or array
            New velocity in km/s.
        """

        # Update the square slowness according to new value
        if isinstance(vp, np.ndarray):
            if vp.shape == self.vp.shape:
                self.vp.data[:] = vp[:]
            else:
                initialize_function(self._vp, vp, self.nbpml)
        else:
            self._vp.data = vp
        self._max_vp = np.max(vp)

    @property
    def m(self):
        return 1 / (self.vp * self.vp)

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
