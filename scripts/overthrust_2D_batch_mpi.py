import sys, os
sys.path.insert(0, '/app/tti')
import numpy as np
from model import Model
from sources import RickerSource, TimeAxis, Receiver
from tti_propagators import TTIPropagators
import segyio as so
from scipy import interpolate, ndimage
from AzureUtilities import read_h5_model, write_h5_model, butter_bandpass_filter, butter_lowpass_filter, resample, process_summaries, read_coordinates, save_rec
from mpi4py import MPI
from devito import configuration
configuration['mpi'] = True

##########################################################################################

# Get runtime arguments and environment variables
if len(sys.argv) > 1:
    run_id = int(str(sys.argv[1]))
else:
    run_id = 0

logging = os.environ["DEVITO_LOGGING"]
num_threads = int(os.environ["OMP_NUM_THREADS"])
rootpath = os.environ["AZ_BATCH_NODE_SHARED_DIR"]
space_order = int(os.environ["SPACE_ORDER"])
batchsize = int(os.environ["BATCHSIZE"])

#########################################################################################

# Read models
rho = read_h5_model(rootpath + '/overthrust/models/rho_with_salt_2D.h5')
epsilon = read_h5_model(rootpath + '/overthrust/models/epsilon_with_salt_2D.h5')
delta = read_h5_model(rootpath + '/overthrust/models/delta_with_salt_2D.h5')
theta = read_h5_model(rootpath + '/overthrust/models/theta_with_salt_2D.h5')
phi = read_h5_model(rootpath + '/overthrust/models/phi_with_salt_2D.h5')
m0 = read_h5_model(rootpath + '/overthrust/models/migration_velocity_2D.h5')
dm = read_h5_model(rootpath + '/overthrust/models/perturbation_2D.h5')

# Read geometry
source_indices = np.load(rootpath + '/overthrust/geometry/source_indices.npy', allow_pickle=True)
idx = source_indices[run_id]
print('Process source no. ',idx , ' out of ', batchsize)

shape = (801, 267)
origin = (0.0, 0.0)
spacing = (12.5, 12.5)

model = Model(shape=shape, origin=origin, spacing=spacing, vp=np.sqrt(1/m0), space_order=space_order,
              epsilon=epsilon, delta=delta, theta=phi, rho=rho, nbpml=40, dm=dm)

comm = model.grid.distributor.comm
size = comm.Get_size()
rank = comm.Get_rank()

#########################################################################################

# Time axis
t0 = 0.
tn = 2800.
dt_shot = 4.
nt = int(tn/dt_shot + 1)
dt = model.critical_dt*.9
time = TimeAxis(start=t0, step=dt, stop=tn)

#########################################################################################

# Source coordinates
file_src = rootpath + '/overthrust/geometry/src_coordinates.h5'
xsrc, ysrc, zsrc = read_coordinates(file_src)
src_coordinates = np.array([xsrc[idx], zsrc[idx]])
src = RickerSource(name='src', grid=model.grid, f0=0.02, time_range=time, npoint=1)
src.coordinates.data[0, 0] = src_coordinates[0]
src.coordinates.data[0, 1] = src_coordinates[1]

# Wavelet
wavelet = np.load(rootpath + '/overthrust/wavelet/wavelet_2D.npy', allow_pickle=True)
src.data[:] = wavelet

# Receivers
nrec = 799
xrec  = np.linspace(12.5, 9987.5, nrec)
zrec = np.linspace(6., 6., nrec)
rec_coords = np.concatenate((xrec.reshape(-1,1), zrec.reshape(-1,1)), axis=1)

#########################################################################################

# Devito operator
tti = TTIPropagators(model, space_order=space_order)

# Shot and gradient
d_obs, u0, v0, summary1 = tti.born(src, rec_coords, save=True, sub=(12, 1))
grad, summary2 = tti.gradient(d_obs, u0, v0, sub=(12, 1), isic=True)
grad.data[:,0:66] = 0   # mute water column

# Gather gradient
if rank > 0:
    # Send rtm image to master
    comm.send(model.vp.local_indices, dest=0, tag=10)
    comm.send(grad.data, dest=0, tag=11)

else:   # Master
    # Initialize full arrays
    rtm = np.empty(shape=model.vp.shape_global, dtype='float32')
    rtm[model.vp.local_indices] = grad.data

    # Collect gradients
    for j in range(1, size):
        local_indices = comm.recv(source=j, tag=10)
        glocal = comm.recv(source=j, tag=11)
        rtm[local_indices] = glocal

    # Remove pml and extent back to full size
    rtm = rtm[model.nbpml:-model.nbpml, model.nbpml:-model.nbpml]  # remove padding

    # Save to bucket
    write_h5_model(rootpath + '/overthrust/rtm/overthrust_2D_rtm_shot_' + str(idx) + '.h5', 'rtm', rtm)

    # Process devito summaries and save in blob storage
    kernel, gflopss, gpointss, oi, ops = process_summaries([summary1, summary2])
    timings = np.array([kernel, gflopss, gpointss, oi, ops])
    timings.dump(rootpath + '/overthrust/timings/timings_rtm_2D_shot_' + str(idx) + '.npy')

# Save shot
filename = rootpath + '/overthrust/data/overthrust_2D_born_data_' + str(idx)
save_rec(d_obs, src_coordinates, filename, nt)
