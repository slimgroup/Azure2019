from __future__ import print_function
import sys
sys.path.insert(0, '/app/devito_isotropic')

import numpy as np
from argparse import ArgumentParser
import psutil, os, gc, random, string, time, subprocess
from PySource import RickerSource, Receiver
from PyModel import Model
from utils import AcquisitionGeometry
from JAcoustic_codegen import forward_modeling, adjoint_born, forward_born
from AzureUtilities import *
from devito.builtins import norm
from scipy import ndimage
from mpi4py import MPI
from devito import configuration
configuration['mpi'] = True

# Read runtime arguments
if len(sys.argv) > 1:
    shot_id = int(sys.argv[1])

print("Source index: ", shot_id)

modeldir = os.environ["MODEL_PATH"]
datadir = os.environ["DATA_PATH"]
gradientdir = os.environ["GRADIENT_PATH"]

# Start timer
tstart = time.time()
chars = string.ascii_uppercase + string.ascii_lowercase + string.digits

# Get environment variables
iteration = 1
num_shots = 1
batchsize = 1
num_chunks = 1
bucket = 'seismic'
queue_names = 'test_queue'

data_path = 'data/'
data_name = 'bp_observed_data_1005.segy'

model_path = 'models/'
velocity_name = 'velocity_model'
water_name = 'water_model'

variable_path = 'variables/'
variable_name = 'test_image_1'
step_length = 1
step_scaling = 1

partial_gradient_path = 'partial_gradient/'
full_gradient_path = 'full_gradient/'
gradient_name = 'test_gradient_1'

# Fetch models from S3
#m0, origin, spacing = model_get(bucket, model_path + velocity_name)
#water = model_get(bucket, model_path + water_name)[0]
m0, origin, spacing = model_read('seismic/models/velocity_model.h5')
water = model_read('seismic/models/water_bottom.h5')[0]

shape = m0.shape
ndims = len(spacing)

# Fetch observed data
idx = 1005   #random.randint(1, num_shots)  # fix to certain bsize for constant?
print("Process shot no.: ", idx)
dorig, sx, sz, gx, gz, tn, dt, nt = segy_read('seismic/data/bp_observed_data_1005.segy')

# Load previous iterations
if iteration == 1:
    x = np.zeros(shape=shape, dtype='float32')
else:
    x = array_get(bucket, variable_path + 'chunk_1/' + variable_name + str(iteration-1))
    if num_chunks > 1:
        for chunk in range(1,num_chunks):
            x_chunk = array_get(bucket, variable_path + 'chunk_' + str(chunk+1) + '/' + variable_name + str(iteration-1))
            x = np.concatenate((x, x_chunk), axis=0)
    x = x.reshape(shape[0], shape[1], order='F')

# Set up model structures
model = Model(shape=shape, origin=origin, spacing=spacing, vp=np.sqrt(1/m0))
comm = model.grid.distributor.comm
size = comm.Get_size()
rank = comm.Get_rank()
print("Shape: ", model.shape)

# Time axis
t0 = 0.
dt_comp = model.critical_dt
nt_comp = int(1 + (tn-t0) / dt_comp) + 1
time_comp = np.linspace(t0, tn, nt_comp)

# Source
f0 = 0.020
src_coordinates = np.empty((1, ndims))
src_coordinates[0, 0] = sx
src_coordinates[0, 1] = sz

# Receiver for predicted data
nrec = len(gx)
rec_coordinates = np.empty((nrec, ndims))
rec_coordinates[:, 0] = gx
rec_coordinates[:, 1] = gz

geometry = AcquisitionGeometry(model, rec_coordinates, src_coordinates, t0=0.0, tn=tn, src_type='Ricker', f0=f0)

# Resample input data to computational grid
dorig = resample(dorig, t0, tn, nt, nt_comp)
geometry.rec.data[:] = dorig

dobs = Receiver(name='rec_t', grid=model.grid, ntime=nt_comp, coordinates=rec_coordinates)
dobs.data[:] = dorig

# Predicted data
if iteration > 1:
    dpred = forward_born(model, geometry)
    sub_rec(dpred, dobs)
else:
    dpred = Receiver(name='rec', grid=model.grid, ntime=nt_comp, coordinates=rec_coordinates)
    dpred.data[:] = -dorig

# Function value
fval = np.array([.5*norm(dpred)**2], dtype='float32')
print("fval, dpred.shape, dobs.shape: ", fval, dpred.shape, dobs.shape)

# Wavefields in memory
t1 = time.time()
opF, u0 = forward_modeling(model, geometry, save=True, u_return=True, op_return=True, tsub_factor=16)
g, summary1, summary2 = adjoint_born(model, dpred, u=u0, is_residual=True, op_forward=opF, tsub_factor=16)
t2 = time.time()
print("Save in memory. Time [s]: ", t2 - t1)

# Gather gradient
if rank > 0:
    # Send result to master
    comm.send(model.m.local_indices, dest=0, tag=10)
    comm.send(g, dest=0, tag=11)

else:   # Master
    # Initialize full array
    gfull = np.empty(shape=model.m.shape_global, dtype='float32')
    gfull[model.m.local_indices] = g

    # Collect gradients
    for j in range(1, size):
        local_indices = comm.recv(source=j, tag=10)
        glocal = comm.recv(source=j, tag=11)
        gfull[local_indices] = glocal

    # Remove pml and extent back to full size
    gfull = gfull[model.nbpml:-model.nbpml, model.nbpml:-model.nbpml]  # remove padding
    gfull = np.reshape(gfull, -1, order='F')

    # Chunk up gradient and write to bucket. Add gradients to SQS queue
    chunk_size = get_chunk_size(len(gfull), num_chunks)
    idx_count = 0
    for j in range(num_chunks):

        # Save to bucket
        file_ext = '0'
        key = partial_gradient_path + 'chunk_' + str(j+1) + '/' + gradient_name
        gwrite = gfull[idx_count:idx_count + chunk_size[j]]
        array_put(gwrite, bucket, key)
        idx_count += chunk_size[j]

        # # Add to queue
        # queue_name = queue_names + str(j+1)
        # print("Queue name: ", queue_name)
        # msg = bucket + '&' + partial_gradient_path + '&' + full_gradient_path + '&' + \
        #     gradient_name + '&' + file_ext + '&' + str(iteration) + '&1&' + str(batchsize) + \
        #     '&' + str(j+1) + '&' + queue_name + '&' + variable_path + '&' + variable_name + \
        #     '&' + str(step_length) + '&' + str(step_scaling)
        #
        # text_file = open("/efs/scratch/devito/sqs_command_" + str(j), "w")
        # text_file.write(msg)
        # text_file.close()

tfinal = time.time()
print("Time spent running script: ", tfinal - tstart)

# Save timings in bucket
t_devito = t2 - t1
t_script = tfinal - tstart

# Kernel time master
t_kernel = 0
for key in summary1:
    t_kernel += summary1[key].time
for key in summary2:
    t_kernel += summary2[key].time

timings = np.array([t_kernel, t_devito, t_script], dtype='float32')
print("Timings: ", timings)
#filename = 'timings/mpi_scaling_nodes_' + str(size)
#array_put(timings, bucket, filename)
