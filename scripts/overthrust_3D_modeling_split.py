import numpy as np
import sys, os
sys.path.insert(0, '/app/tti')
from model import Model
from sources import RickerSource, TimeAxis, Receiver
from tti_propagators import TTIPropagators
import segyio as so
from scipy import interpolate, ndimage
from AzureUtilities import read_h5_model, write_h5_model, butter_bandpass_filter, butter_lowpass_filter, resample, process_summaries, read_coordinates, save_rec, restrict_model_to_receiver_grid, extent_gradient
from scipy import interpolate, ndimage
from mpi4py import MPI
from devito import configuration
configuration['mpi'] = True

##########################################################################################

def limit_receiver_grid(xsrc, ysrc, xrec, yrec, zrec, maxoffset):

    xmin = np.max([xsrc - maxoffset, 12.5])
    xmax = np.min([xsrc + maxoffset, 9987.5])
    ymin = np.max([ysrc - maxoffset, 12.5])
    ymax = np.min([ysrc + maxoffset, 9987.5])

    print('xrange: ', xmin, ' to ', xmax)
    print('yrange: ', ymin, ' to ', ymax)

    xnew = []
    ynew = []
    znew = []

    for j in range(len(xrec)):
        if xrec[j] >= xmin and xrec[j] <= xmax and yrec[j] >= ymin and yrec[j] <= ymax:
            xnew.append(xrec[j])
            ynew.append(yrec[j])
            znew.append(zrec[j])

    xnew = np.array(xnew)
    ynew = np.array(ynew)
    znew = np.array(znew)

    return xnew, ynew, znew

def check_thomsen_parameters(model):
    err = np.where(model.delta.data > model.epsilon.data)
    num_err = np.sum([len(err[0]), len(err[1]), len(err[2])])
    if num_err > 0:
        print('Warning: delta > epsilon')

##########################################################################################

# Get runtime arguments and environment variables
if len(sys.argv) > 1:
    run_id = int(str(sys.argv[1]))
else:
    run_id = 0

logging = os.environ["DEVITO_LOGGING"]
num_threads = int(os.environ["OMP_NUM_THREADS"])
rootpath = os.environ["AZ_BATCH_NODE_SHARED_DIR"]

container = os.environ["BLOB_CONTAINER"]    # overthrust
space_order = int(os.environ["SPACE_ORDER"])
batchsize = int(os.environ["BATCHSIZE"])

#########################################################################################

# Read models
rho = read_h5_model(rootpath + '/overthrust/models/rho_with_salt.h5')
epsilon = read_h5_model(rootpath + '/overthrust/models/epsilon_with_salt.h5')
delta = read_h5_model(rootpath + '/overthrust/models/delta_with_salt.h5')
theta = read_h5_model(rootpath + '/overthrust/models/theta_with_salt.h5')
phi = read_h5_model(rootpath + '/overthrust/models/phi_with_salt.h5')
m0 = read_h5_model(rootpath + '/overthrust/models/migration_velocity.h5')
dm = read_h5_model(rootpath + '/overthrust/models/perturbation.h5')
dm[:, :, 0:29] = 0.0    # Extend water column
delta[delta >= epsilon] = .9 * epsilon[delta >= epsilon]    # fix thomsen parameters

# Read geometry
source_indices = np.load(rootpath + '/overthrust/geometry/source_indices.npy', allow_pickle=True)
idx = source_indices[run_id]
print('Process source no. ', idx, ' out of ', batchsize)

shape_full = (801, 801, 267)
origin_full = (0.0, 0.0, 0.0)
spacing = (12.5, 12.5, 12.5)

# Read coordinates and source index
file_src = rootpath + '/overthrust/geometry/src_coordinates.h5'
file_rec = rootpath + '/overthrust/geometry/rec_coordinates.h5'
xsrc, ysrc, zsrc = read_coordinates(file_src)
xrec, yrec, zrec = read_coordinates(file_rec)
xsrc = xsrc[idx]; ysrc = ysrc[idx]; zsrc = zsrc[idx]
print('Source location: ', xsrc, ', ', ysrc, ', ', zsrc)

# Limit receiver grid
buffersize = 500    # in m
maxoffset = 3787.5  # x/y direction in m
xrec, yrec, zrec = limit_receiver_grid(xsrc, ysrc, xrec, yrec, zrec, maxoffset)

# Restrict models to receiver grid
print('Original shape: ', shape_full, ' and origin: ', origin_full)
m0, shape, origin = restrict_model_to_receiver_grid(xsrc, xrec, m0, spacing, origin_full, buffer_size=buffersize, sy=ysrc, gy=yrec)
rho = restrict_model_to_receiver_grid(xsrc, xrec, rho, spacing, origin_full, buffer_size=buffersize, sy=ysrc, gy=yrec)[0]
epsilon = restrict_model_to_receiver_grid(xsrc, xrec, epsilon, spacing, origin_full, buffer_size=buffersize, sy=ysrc, gy=yrec)[0]
delta = restrict_model_to_receiver_grid(xsrc, xrec, delta, spacing, origin_full, buffer_size=buffersize, sy=ysrc, gy=yrec)[0]
theta = restrict_model_to_receiver_grid(xsrc, xrec, theta, spacing, origin_full, buffer_size=buffersize, sy=ysrc, gy=yrec)[0]
phi = restrict_model_to_receiver_grid(xsrc, xrec, phi, spacing, origin_full, buffer_size=buffersize, sy=ysrc, gy=yrec)[0]
dm = restrict_model_to_receiver_grid(xsrc, xrec, dm, spacing, origin_full, buffer_size=buffersize, sy=ysrc, gy=yrec)[0]
print('New shape: ', shape, ' and origin ', origin)

model = Model(shape=shape, origin=origin, spacing=spacing, vp=np.sqrt(1/m0), space_order=space_order,
              epsilon=epsilon, delta=delta, theta=theta, phi=phi, rho=rho, nbpml=40, dm=dm, dt=0.64)

check_thomsen_parameters(model)

comm = model.grid.distributor.comm
size = comm.Get_size()
rank = comm.Get_rank()

#########################################################################################

# Time axis
t0 = 0.
tn = 500.
dt_shot = 4.
nt = int(tn/dt_shot + 1)
dt = 0.64   #model.critical_dt*.8
time = TimeAxis(start=t0, step=dt, stop=tn)

#########################################################################################


rec_coords = np.concatenate((xrec.reshape(-1,1), yrec.reshape(-1,1), zrec.reshape(-1,1)), axis=1)
print('Number of receivers: ', rec_coords.shape[0])

# Source coordinates
src_coordinates = np.array([xsrc, ysrc, zsrc])
src = RickerSource(name='src', grid=model.grid, f0=0.02, time_range=time, npoint=1)
src.coordinates.data[0, 0] = src_coordinates[0]
src.coordinates.data[0, 1] = src_coordinates[1]
src.coordinates.data[0, 2] = src_coordinates[2]


#########################################################################################

# Devito operator
tti = TTIPropagators(model, space_order=space_order)

# Data and RTM
d_obs, u0, v0, summary1 = tti.forward(src, rec_coords, autotune=('aggressive', 'runtime'))

# Save shot
filename = rootpath + '/overthrust/data/overthrust_3D_data_source_' + str(idx)
save_rec(d_obs, src_coordinates, filename, nt)
