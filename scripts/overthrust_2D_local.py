import sys, os

# Assume JUDI is installed at ~/.julia/dev/JUDI
sys.path.insert(0,'/home/pwitte/JUDI.jl/src/pysource/')
import numpy as np
import matplotlib.pyplot as plt
from models import Model
from sources import RickerSource, TimeAxis, Receiver
from propagators import born, gradient, forward
import segyio as so
from scipy import interpolate, ndimage
from AzureUtilities import read_h5_model, write_h5_model, butter_bandpass_filter, butter_lowpass_filter, resample, process_summaries, read_coordinates, save_rec, restrict_model_to_receiver_grid, extent_gradient
from scipy import interpolate, ndimage

#########################################################################################

def limit_receiver_grid(xsrc, xrec, zrec, maxoffset):

    xmin = np.max([xsrc - maxoffset, 12.5])
    xmax = np.min([xsrc + maxoffset, 9987.5])

    print('xrange: ', xmin, ' to ', xmax)

    xnew = []
    znew = []

    for j in range(len(xrec)):
        if xrec[j] >= xmin and xrec[j] <= xmax:
            xnew.append(xrec[j])
            znew.append(zrec[j])

    xnew = np.array(xnew)
    znew = np.array(znew)

    return xnew, znew

#########################################################################################

# Read models
rho = read_h5_model('../data/models/rho_with_salt_2D.h5')
epsilon = read_h5_model('../data/models/epsilon_with_salt_2D.h5')
delta = read_h5_model('../data/models/delta_with_salt_2D.h5')
theta = read_h5_model('../data/models/theta_with_salt_2D.h5')
m0 = read_h5_model('../data/models/migration_velocity_2D.h5')
dm = read_h5_model('../data/models/perturbation_2D.h5')

# Set dm to zero in water
dm[:,0:29] = 0.

shape_full = (801, 267)
origin_full = (0.0, 0.0)
spacing = (12.5, 12.5)
so = 12

# Read source coordinates
shot_no = 0
file_idx = '../data/geometry/source_indices.npy'
file_src = '../data/geometry/src_coordinates.h5'
xsrc_full, ysrc_full, zsrc_full = read_coordinates(file_src)
idx = np.load(file_idx, allow_pickle=True)[shot_no]

xsrc = 7000.
zsrc = 300 - 12.5

# Receivers coordinates
nrec = 799
xrec_full = np.array(np.linspace(12.5, 9987.5, nrec))
zrec_full = np.array(np.linspace(6., 6., nrec))

# Limit receiver grid
buffersize = 500    # in m
maxoffset = 3787.5  # in m
xrec, zrec = limit_receiver_grid(xsrc, xrec_full, zrec_full, maxoffset)

# Restrict models
print('Original shape: ', shape_full, ' and origin ', origin_full)
m0, shape, origin = restrict_model_to_receiver_grid(xsrc, xrec, m0, spacing, origin_full, buffer_size=buffersize)
rho = restrict_model_to_receiver_grid(xsrc, xrec, rho, spacing, origin_full, buffer_size=buffersize)[0]
epsilon = restrict_model_to_receiver_grid(xsrc, xrec, epsilon, spacing, origin_full, buffer_size=buffersize)[0]
delta = restrict_model_to_receiver_grid(xsrc, xrec, delta, spacing, origin_full, buffer_size=buffersize)[0]
theta = restrict_model_to_receiver_grid(xsrc, xrec, theta, spacing, origin_full, buffer_size=buffersize)[0]
dm = restrict_model_to_receiver_grid(xsrc, xrec, dm, spacing, origin_full, buffer_size=buffersize)[0]
print('New shape: ', shape, ' and origin ', origin)

# Model structure
model = Model(shape=shape, origin=origin, spacing=spacing, vp=np.sqrt(1/m0), space_order=so,
              epsilon=epsilon, delta=delta, theta=theta, rho=rho, nbpml=40, dm=dm)

#########################################################################################

# Time axis
t0 = 0.
tn = 2500.
dt_shot = 0.65  # model.critical_dt
nt = int(tn/dt_shot + 1)
time = np.linspace(0, tn, nt)

#########################################################################################

# Coordinates
src = RickerSource(name='src', grid=model.grid, f0=.015, time=time, npoint=1)
src.coordinates.data[0, 0] = xsrc
src.coordinates.data[0, 1] = zsrc

# nrec = len(xrec)
# rec_t = Receiver(name='rec_t', grid=model.grid, npoint=nrec, ntime=nt)
# rec_t.coordinates.data[:, 0] = xrec
# rec_t.coordinates.data[:, 1] = zrec

nrec = len(xrec)
rec_coords = np.empty((nrec, 2))
rec_coords[:, 0] = xrec
rec_coords[:, 1] = zrec


#########################################################################################

def resample(rec, num, time):
    #start, stop = rec._time_range.start, rec._time_range.stop
    #dt0 = rec._time_range.step
    start = time[0]
    stop = time[-1]
    new_time_range = TimeAxis(start=start, stop=stop, num=num)
    dt = new_time_range.step
    to_interp = np.asarray(rec.data)
    data = np.zeros((num, to_interp.shape[1]))
    for i in range(to_interp.shape[1]):
        tck = interpolate.splrep(time, to_interp[:, i], k=3)
        data[:, i] = interpolate.splev(new_time_range.time_values, tck)
    coords_loc = np.asarray(rec.coordinates.data)
    # Return new object
    return data, coords_loc

# Devito operator
d_obs, u0, summary1 = forward(model, src.coordinates.data, rec_coords, src.data, save=True, t_sub=12)
grad, summary2 = gradient(model, d_obs, rec_coords, u0, isic=True)

grad.data[:,0:66] = 0   # mute water column

# Remove pml and pad
rtm = grad.data[model.nbl:-model.nbl, model.nbl:-model.nbl]  # remove padding
rtm = extent_gradient(shape_full, origin_full, shape, origin, spacing, rtm)
plt.figure(); plt.imshow(d_obs.data, vmin=-1e-1, vmax=1e-1, cmap='gray', aspect='auto')
plt.figure(); plt.imshow(np.transpose(rtm), vmin=-2e0, vmax=2e0, cmap='gray', aspect='auto')
plt.show()
