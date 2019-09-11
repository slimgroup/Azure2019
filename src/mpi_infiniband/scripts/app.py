# Test BP TTI Reverse-Time-Migration
# Georgia Institute of Technology
# Date: 9/10/2019

import sys, os, random, string, time, subprocess
import numpy as np
import segyio, time
from scipy import ndimage, interpolate
from devito import *
from devito.logger import info
from tti_propagators import *
from model import *
from examples.seismic import RickerSource, Receiver, TimeAxis
from azure.storage.queue import QueueService, QueueMessageFormat
from AzureUtilities import segy_model_read, segy_read, array_put, restrict_model_to_receiver_grid, extent_gradient, resample, process_summaries
from mpi4py import MPI

# Queue credentials
queue_service = QueueService(account_name='', account_key='')
queue_service.encode_function = QueueMessageFormat.text_base64encode

##########################################################################################

# Get runtime arguments and environment variables
if len(sys.argv) > 1:
    run_id = int(str(sys.argv[1]))

logging = os.environ["DEVITO_LOGGING"]
num_threads = int(os.environ["OMP_NUM_THREADS"])
rootpath = os.environ["AZ_BATCH_NODE_SHARED_DIR"]

container = os.environ["BLOB_CONTAINER"]
space_order =  int(os.environ["SPACE_ORDER"])
restrict = os.environ["RESTRICT_MODEL"]

batchsize = int(os.environ["BATCHSIZE"])
iteration = int(os.environ["ITERATION"])
maxiter = int(os.environ["MAXITER"])

partial_path = os.environ["PARTIAL_GRAD_PATH"]
full_path = os.environ["FULL_GRAD_PATH"]
grad_name = os.environ["GRAD_NAME"]

print('Process sourc no. ', run_id, ' out of ', batchsize)

# Read model parameters
vp = segy_model_read(rootpath + '/seismic/models/Vp_Model.sgy')[0] / 1e3  # velocity in km/s
epsilon = segy_model_read(rootpath + '/seismic/models/Epsilon_Model.sgy')[0]
delta = segy_model_read(rootpath + '/seismic/models/Delta_Model.sgy')[0]
theta = segy_model_read(rootpath + '/seismic/models/Theta_Model.sgy')[0]
theta *= -np.pi/180   # convert to rad

# Read shot
filename = 'BPTTI_' + str(run_id) + '.segy'
dorig, sx, sz, gx, gz, tn, dt, nt = segy_read(rootpath + '/seismic/data/' + filename)

# Model parameters
shape_full = vp.shape
origin_full = (0., 0.)
spacing = (6.25, 6.25)
ndims = len(spacing)
dt_full = np.float32('%.3f' % (.38 * np.min(spacing) / (np.sqrt(np.max(1 + 2 * epsilon))*np.max(vp))))

# Restrict model to receiver area
if restrict == 'TRUE':
    vp, shape, origin = restrict_model_to_receiver_grid(sx, gx, vp, spacing, origin_full)
    epsilon = restrict_model_to_receiver_grid(sx, gx, epsilon, spacing, origin_full)[0]
    delta = restrict_model_to_receiver_grid(sx, gx, delta, spacing, origin_full)[0]
    theta = restrict_model_to_receiver_grid(sx, gx, theta, spacing, origin_full)[0]
else:
    origin = origin_full
    shape = shape_full

# Model structures and MPI environment
model = Model(shape=shape, origin=origin, spacing=spacing, vp=vp,
              epsilon=epsilon, delta=delta, theta=theta, nbpml=40, dt=dt_full)

comm = model.grid.distributor.comm
size = comm.Get_size()
rank = comm.Get_rank()

# Time axis
t0 = 0.
dt_comp = model.critical_dt
nt_comp = int(1 + (tn-t0) / dt_comp) + 1
time_comp = TimeAxis(start=t0, step=dt_comp, stop=tn)

# Source wavelet
f0 = 0.025
src_coordinates = np.empty((1, ndims))
src_coordinates[0, 0] = sx
src_coordinates[0, 1] = sz
src = RickerSource(name='src', grid=model.grid, f0=0.025, time_range=time_comp, npoint=1, coordinates=src_coordinates)

# Receiver for predicted data
nrec = len(gx)
rec_coordinates = np.empty((nrec, ndims))
rec_coordinates[:, 0] = gx
rec_coordinates[:, 1] = gz
rec = Receiver(name='rec', grid=model.grid, npoint=nrec, time_range=time_comp, coordinates=rec_coordinates)
dorig = resample(dorig, t0, tn, nt, nt_comp)
rec.data[:, :] = dorig

# RTM
tti = TTIPropagators(model, space_order=space_order)
u0, v0, summary1 = tti.forward(src, None, save=True, sub=(16, 1), norec=True)[1:]
g, summary2 = tti.gradient(rec, u0, v0, sub=(16, 1))

# Gather gradient
if rank > 0:
    # Send result to master
    comm.send(model.m.local_indices, dest=0, tag=10)
    comm.send(g.data, dest=0, tag=11)

else:   # Master
    # Initialize full array
    gfull = np.empty(shape=model.m.shape_global, dtype='float32')
    gfull[model.m.local_indices] = g.data

    # Collect gradients
    for j in range(1, size):
        local_indices = comm.recv(source=j, tag=10)
        glocal = comm.recv(source=j, tag=11)
        gfull[local_indices] = glocal

    # Remove pml and extent back to full size
    gfull = gfull[model.nbpml:-model.nbpml, model.nbpml:-model.nbpml]  # remove padding
    if restrict == 'TRUE':
        gfull = extent_gradient(shape_full, origin_full, shape, origin, spacing, gfull) # extend back to full size
    gfull = np.reshape(gfull, -1, order='F')

    # Save to bucket
    array_put(gfull, container, partial_path + grad_name + str(run_id))

    # Send msg to queue
    msg = container + '&' + partial_path + '&' + full_path + '&' + grad_name + '&' + str(run_id) + '&' + str(iteration) + '&' + str(maxiter) + '&1&' + str(batchsize)
    print('Out message: ', msg, '\n')
    queue_service.put_message('gradientqueue', msg)

    # Process devito summaries and save in blob storage
    kernel, gflopss, gpointss, oi, ops = process_summaries([summary1, summary2])
    array_put(np.array([kernel, gflopss, gpointss, oi, ops]), container, 'timings/' + grad_name + str(run_id))
