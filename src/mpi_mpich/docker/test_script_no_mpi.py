from __future__ import print_function
import sys
sys.path.insert(0, '/app/devito-tti')

import numpy as np
import psutil, os, gc, random, string, time
from PySource import RickerSource, Receiver
from PyModel import Model
from utils import AcquisitionGeometry
from JAcoustic_codegen import forward_modeling, adjoint_born, forward_born, forward_freq_modeling, adjoint_freq_born
from AzureUtilities import *
from scipy import ndimage
from devito import Eq, Operator, TimeFunction
from devito import mode_performance, configuration

#mode_performance()
#configuration['autotuning '] = ('aggressive', 'runtime')

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
space_order=8

# Fetch models from S3
m0, origin_full, spacing = model_get(bucket, model_path + velocity_name)
water = model_get(bucket, model_path + water_name)[0]
shape_full = m0.shape
ndims = len(spacing)

idx = 800   #shot[node_idx]
print("Process shot no.: ", idx)
dorig, sx, sz, gx, gz, tn, dt, nt = segy_read('seismic/data/bp_observed_data_1005.segy')

# Restrict models to receiver area + buffer (default is 500 m)
m0, shape, origin = restrict_model_to_receiver_grid(sx, gx, m0, spacing, origin_full, buffer_size=5000)
water = restrict_model_to_receiver_grid(sx, gx, water, spacing, origin_full, buffer_size=5000)[0]

print("Original shape: ", shape_full)
print("New shape: ", shape)

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
dobs = Receiver(name='rec_t', grid=model.grid, ntime=nt_comp, coordinates=rec_coordinates)
dobs.data[:] = dorig

# Predicted data
if iteration > 1:
    dpred, summary0 = forward_born(model, geometry.src_positions, geometry.src.data, geometry.rec_positions, space_order=space_order)
    # Residual and function value
    dpred = dpred - dobs
else:
    dpred = Receiver(name='rec', grid=model.grid, ntime=nt_comp, coordinates=rec_coordinates)
    dpred.data[:] = -dorig

# Function value
fval = np.array([.5*np.linalg.norm(dpred.data)**2], dtype='float32')
print("fval, dpred.shape, dobs.shape: ", fval, dpred.shape, dobs.shape)

# Wavefields in memory
t1 = time.time()
opF, u0 = forward_modeling(model, geometry.src_positions, geometry.src.data, geometry.rec_positions, save=True, u_return=True, op_return=True, tsub_factor=16, space_order=space_order)
g, summary1, summary2 = adjoint_born(model, geometry.rec_positions, dpred.data[:], u=u0, is_residual=True, op_forward=opF, tsub_factor=16, space_order=space_order)
t2 = time.time()
print("Save in memory. Time [s]: ", t2 - t1)

# Remove pml and extent back to full size
g = g[model.nbpml:-model.nbpml, model.nbpml:-model.nbpml]  # remove padding
g = extent_gradient(shape_full, origin_full, shape, origin, spacing, g)
#g = np.reshape(g, -1, order='F')

# Chunk up gradient and write to bucket. Add gradients to SQS queue
chunk_size = get_chunk_size(len(g), num_chunks)
idx_count = 0
for j in range(num_chunks):

    # Save to bucket
    file_ext = ''.join(random.sample(chars*12, 12))
    key = partial_gradient_path + 'chunk_' + str(j+1) + '/' + gradient_name
    gwrite = g[idx_count:idx_count + chunk_size[j]]
    array_put(gwrite, bucket, key)
    idx_count += chunk_size[j]

    # Add to queue
    # queue_name = queue_names + str(j+1)
    # msg = bucket + '&' + partial_gradient_path + '&' + full_gradient_path + '&' + \
    #     gradient_name + '&' + file_ext + '&' + str(iteration) + '&1&' + str(batchsize) + \
    #     '&' + str(j+1) + '&' + queue_name + '&' + variable_path + '&' + variable_name + \
    #     '&' + str(step_length) + '&' + str(step_scaling)
    #
    # # SQS url
    # url = sqs_client.list_queues(QueueNamePrefix=queue_name)['QueueUrls'][0]
    # idx = url.find('com') + 4
    # user_id = url[idx:idx+12]
    # url_address = 'https://sqs.us-east-1.amazonaws.com/' + user_id + '/' + queue_name
    #
    # # Send message
    # sqs_client.send_message(QueueUrl=url_address, DelaySeconds=0, MessageBody=msg)

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
filename = 'timings/mpi_scaling_nodes_1'
array_put(timings, bucket, filename)
