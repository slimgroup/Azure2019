import sys, os
# Add path to TTI module
sys.path.insert(0, os.getcwd()[:-8] + '/src/AzureBatch/docker/tti_image/tti')
import numpy as np
import matplotlib.pyplot as plt
from model import Model
from sources import RickerSource, TimeAxis, Receiver
from tti_propagators import TTIPropagators
import segyio as so
from scipy import interpolate, ndimage
from AzureUtilities import read_h5_model, butter_bandpass_filter, butter_lowpass_filter, resample, process_summaries, read_coordinates, restrict_model_to_receiver_grid
from scipy import interpolate, ndimage

#########################################################################################

# Shot number (get from batch)
shot_no = 0

# Read models
rho = read_h5_model('../data/models/rho_with_salt.h5')
epsilon = read_h5_model('../data/models/epsilon_with_salt.h5')
delta = read_h5_model('../data/models/delta_with_salt.h5')
theta = read_h5_model('../data/models/theta_with_salt.h5')
phi = read_h5_model('../data/models/phi_with_salt.h5')
m0 = read_h5_model('../data/models/migration_velocity.h5')
dm = read_h5_model('../data/models/perturbation.h5')
delta[delta >= epsilon] = .5 * epsilon[delta >= epsilon]

shape = (801, 801, 267)
origin_full = (0.0, 0.0, 0.0)
spacing = (12.5, 12.5, 12.5)
so = 8

# Read coordinates
file_idx = '../data/geometry/source_indices.npy'
file_src = '../data/geometry/src_coordinates.h5'
file_rec = '../data/geometry/rec_coordinates.h5'
xsrc_full, ysrc_full, zsrc_full = read_coordinates(file_src)
xrec_full, yrec_full, zrec_full = read_coordinates(file_rec)
idx = np.load(file_idx)[shot_no]
xsrc = xsrc_full[idx]; ysrc = ysrc_full[idx]; zsrc = zsrc_full[idx]

buffersize = 500    # in m
maxoffset = 13987.5  # in m

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
    err = np.where(model.delta.data >= model.epsilon.data)
    num_err = np.sum([len(err[0]), len(err[1]), len(err[2])])
    if num_err > 0:
        print('Warning: delta >= epsilon')

xrec, yrec, zrec = limit_receiver_grid(xsrc, ysrc, xrec_full, yrec_full, zrec_full, maxoffset)

# Plot grids
plt.figure(); plt.plot(xrec_full, yrec_full, 'ob', xrec, yrec, 'or', xsrc, ysrc, 'xg'); plt.title('Receiver grid')
plt.legend(['Full receiver grid', 'Limited receiver grid', 'Source'])
plt.figure(); plt.plot(xsrc_full, ysrc_full, 'ob', xsrc, ysrc, 'xg'); plt.title('Source grid')
print('Original shape: ', m0.shape)
#plt.show()

# Restrict models
m0, shape, origin = restrict_model_to_receiver_grid(xsrc, xrec, m0, spacing, origin_full, buffer_size=buffersize, sy=ysrc, gy=yrec)
rho = restrict_model_to_receiver_grid(xsrc, xrec, rho, spacing, origin_full, buffer_size=buffersize, sy=ysrc, gy=yrec)[0]
epsilon = restrict_model_to_receiver_grid(xsrc, xrec, epsilon, spacing, origin_full, buffer_size=buffersize, sy=ysrc, gy=yrec)[0]
delta = restrict_model_to_receiver_grid(xsrc, xrec, delta, spacing, origin_full, buffer_size=buffersize, sy=ysrc, gy=yrec)[0]
theta = restrict_model_to_receiver_grid(xsrc, xrec, theta, spacing, origin_full, buffer_size=buffersize, sy=ysrc, gy=yrec)[0]
phi = restrict_model_to_receiver_grid(xsrc, xrec, phi, spacing, origin_full, buffer_size=buffersize, sy=ysrc, gy=yrec)[0]
dm = restrict_model_to_receiver_grid(xsrc, xrec, dm, spacing, origin_full, buffer_size=buffersize, sy=ysrc, gy=yrec)[0]
print('New shape: ', m0.shape)

# Model structure
model = Model(shape=shape, origin=origin, spacing=spacing, vp=np.sqrt(1/m0), space_order=so,
              epsilon=epsilon, delta=delta, theta=theta, phi=phi, rho=rho, nbpml=40, dm=dm, dt=0.64)


# #########################################################################################

# Time axis
t0 = 0.
tn = 100.
dt_shot = 4.
nt = tn/dt_shot + 1
dt = 0.64   #model.critical_dt*.8
time = TimeAxis(start=t0, step=dt, stop=tn)

#########################################################################################

# Receiver coordinate
rec_coords = np.concatenate((xrec.reshape(-1,1), yrec.reshape(-1,1), zrec.reshape(-1,1)), axis=1)

# Source coordinates
src_coordinates = np.array([xsrc, ysrc, zsrc])
src = RickerSource(name='src', grid=model.grid, f0=0.02, time_range=time, npoint=1)
src.coordinates.data[0, :] = src_coordinates[:]

# Read Wavelet
wavelet = np.load('../data/wavelet/wavelet_3D.npy')
src.data[:] = wavelet[0:len(src.data)]

#########################################################################################

# # Devito operator
tti = TTIPropagators(model, space_order=8)
#
# # Data
# d_obs= tti.born(src, rec_coords)[0]
# d_obs = resample(d_obs.data, t0, tn, time.num, nt)
#
# # Process summary
# kernel, gflopss, gpointss, oi, ops = process_summaries([summary1])
