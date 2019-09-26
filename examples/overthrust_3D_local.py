import numpy as np
import matplotlib.pyplot as plt
from model import Model
from sources import RickerSource, TimeAxis, Receiver
from tti_propagators import TTIPropagators
import segyio as so
from scipy import interpolate, ndimage
from AzureUtilities import read_h5_model, butter_bandpass_filter, butter_lowpass_filter, resample, process_summaries, read_coordinates
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

shape = (801, 801, 267)
origin = (0.0, 0.0, 0.0)
spacing = (12.5, 12.5, 12.5)
so = 12

model = Model(shape=shape, origin=origin, spacing=spacing, vp=np.sqrt(1/m0), space_order=so,
              epsilon=epsilon, delta=delta, theta=phi, rho=rho, nbpml=40, dm=dm)


#########################################################################################

# Time axis
t0 = 0.
tn = 3200.
dt_shot = 4.
nt = tn/dt_shot + 1
dt = model.critical_dt*.9
time = TimeAxis(start=t0, step=dt, stop=tn)

#########################################################################################

# Read coordinates and source index
file_idx = '../data/geometry/source_indices.npy'
file_src = '../data/geometry/src_coordinates.h5'
file_rec = '../data/geometry/rec_coordinates.h5'

xsrc, ysrc, zsrc = read_coordinates(file_src)
xrec, yrec, zrec = read_coordinates(file_rec)
idx = np.load(file_idx)[shot_no]
rec_coords = np.concatenate((xrec.reshape(-1,1), yrec.reshape(-1,1), zrec.reshape(-1,1)), axis=1)
nsrc = len(xsrc)

# Source coordinates
src_coordinates = [xsrc[idx], ysrc[idx], zsrc[idx]]
src = RickerSource(name='src', grid=model.grid, f0=0.02, time_range=time, npoint=1)
src.coordinates.data[0, :] = src_coordinates[:]

# Wavelet
wavelet = np.concatenate((np.zeros((20,)), np.load("../data/wavelet/wavelet.npy"), np.zeros((100,))))
twave = [i*1.2 for i in range(wavelet.shape[0])]
tnew = [i*dt for i in range(int(1 + (twave[-1]-t0) / dt))]
fq = interpolate.interp1d(twave, wavelet, kind='linear')
q_custom = np.zeros(src.data.shape)
q_custom[:len(tnew), 0] = fq(tnew)
q_custom[:, 0] = butter_bandpass_filter(q_custom[:, 0], .01, .03, 1/dt)
q_custom[:, 0] = np.max(src.data) * q_custom[:, 0] / np.max(q_custom[:, 0])
src.data[:, 0] = q_custom[:, 0]

#########################################################################################

# Devito operator
tti = TTIPropagators(model, space_order=12)

# Data
d_obs, u, v, summary1 = tti.born(src, rec_coords)
d_obs = resample(d_obs.data, t0, tn, time.num, nt)

# Process summary
kernel, gflopss, gpointss, oi, ops = process_summaries([summary1])
