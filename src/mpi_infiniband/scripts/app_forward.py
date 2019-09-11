# Test BP TTI

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

run_id = 400
space_order = 8    # os.environ["SPACE_ORDER"]
logging = os.environ["DEVITO_LOGGING"]
num_threads = int(os.environ["OMP_NUM_THREADS"])
rootpath = os.environ["AZ_BATCH_NODE_SHARED_DIR"]

# Read model parameters
vp = segy_model_read(rootpath + '/seismic/models/Vp_Model.sgy')[0] / 1e3  # velocity in mk/s
epsilon = segy_model_read(rootpath + '/seismic/models/Epsilon_Model.sgy')[0]
delta = segy_model_read(rootpath + '/seismic/models/Delta_Model.sgy')[0]
theta = segy_model_read(rootpath + '/seismic/models/Theta_Model.sgy')[0]

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
restrict = True
if restrict is True:
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
print('Full model size: ', shape)
print('Local model size: ', model.shape)

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
rec.data[:, :] = dorig

# Forward modeling (don't save wavefield)
tti = TTIPropagators(model, space_order=space_order)
d_pred = tti.forward(src, rec_coordinates, save=False, autotune=('aggressive', 'runtime'))[0]
print("Size of predicted data: ", d_pred.shape)

# Save predicted shot record
# filename_out = rootpath + '/seismic/data/d_pred.segy'
# segy_write(d_pred.data, sx, sz, gx, gz, dt, filename_out)
