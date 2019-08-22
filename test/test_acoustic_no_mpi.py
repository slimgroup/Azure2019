from __future__ import print_function
import numpy as np
import psutil, os, gc, time, sys
from scipy import ndimage
from PySource import RickerSource, PointSource, Receiver
from PyModel import Model
from utils import AcquisitionGeometry
from JAcoustic_codegen import forward_modeling, adjoint_modeling, forward_born, adjoint_born
import matplotlib.pyplot as plt
from AzureUtilities import segy_write, resample, array_put

if len(sys.argv) > 1:
    run_id = str(sys.argv[1])
else:
    run_id=0

tstart = time.time()

num_cores = os.environ['OMP_NUM_THREADS']
omp_places = os.environ['OMP_PLACES']
subsampling_factor=4

print("Run test with numcores: ", num_cores, " and omp places: ", omp_places)

# Model
shape = (120, 120, 120)
spacing = (10, 10, 10)
origin = (0., 0., 0.)
nrec = 101

# Velocity
v = np.empty(shape, dtype=np.float32)
v[:, :, :55] = 1.5
v[:, :, 55:] = 3.0
v0 = ndimage.gaussian_filter(v, sigma=5)
m = (1./v)**2
m0 = (1./v0)**2
dm = m - m0

# Set up model structures
model = Model(shape=shape, origin=origin, spacing=spacing, vp=v, nbpml=30)
model0 = Model(shape=shape, origin=origin, spacing=spacing, vp=v0, dm=dm,nbpml=30)

# Time axis
t0 = 0.
tn = 1200.
num_wavefields = int(tn/4)
mem = ((model.shape[0] + 2*model.nbpml) * (model.shape[1] + 2*model.nbpml) * (model.shape[2] + 2*model.nbpml) *num_wavefields/subsampling_factor * 8)/1024**3

print("No. of computational time steps: ", tn/model.critical_dt)
print("No. of time samples at 4 ms: ", num_wavefields)
print("Memory required: ", mem)

# Source
f0 = 0.012
src_coordinates = np.empty((1, len(spacing)))
src_coordinates[0, :] = np.array(model.domain_size) * 0.5
src_coordinates[0,-1] = 20.

# Receiver for observed data
rec_coordinates = np.empty((nrec, len(spacing)))
rec_coordinates[:, 0] = np.linspace(0, model.domain_size[0], num=nrec)
rec_coordinates[:, 1] = np.linspace(0, model.domain_size[1], num=nrec)
rec_coordinates[:, -1] = 20.

geometry = AcquisitionGeometry(model, rec_coordinates, src_coordinates, t0=0.0, tn=tn, src_type='Ricker', f0=f0)
geometry0 = AcquisitionGeometry(model0, rec_coordinates, src_coordinates, t0=0.0, tn=tn, src_type='Ricker', f0=f0)

# Nonlinear modeling
dobs = forward_modeling(model, geometry, save=False, op_return=False)[0]
dpred = forward_modeling(model0, geometry0, save=False, op_return=False)[0]

dpred.data[:] = dpred.data[:] - dobs.data[:]   # residual
#qad = adjoint_modeling(model, geometry)

# Linearized modeling
#dlin = forward_born(model, geometry, isic=False)

# Gradient
opF, u0 = forward_modeling(model0, geometry0, save=True, u_return=True, op_return=True, tsub_factor=2)
g = adjoint_born(model0, dpred, u=u0, op_forward=opF, tsub_factor=2)
plt.imshow(np.transpose(g.data), vmin=-2e-1, vmax=2e-1); plt.show()
