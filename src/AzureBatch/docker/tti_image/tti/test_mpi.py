from devito import *
from sources import RickerSource, TimeAxis
import numpy as np

grid = Grid((100, 100, 100), extent=(1000., 1000., 1000.))
x, y, z = grid.dimensions
u = TimeFunction(name="u", grid=grid, space_order=4, time_order=1)

# Time axis
tstart = 0.
tn = 500.
dt = 1.
time_axis = TimeAxis(start=tstart, step=dt, stop=tn)
src = RickerSource(name='src', grid=grid, f0=0.020, time_range=time_axis, npoint=1)
src.coordinates.data[0, :] = np.array([500., 500., 500.])


Operator([Eq(u.forward, u.biharmonic() + (x + y + z))] + src.inject(u, expr=src))()
