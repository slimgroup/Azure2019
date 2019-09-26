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

# Read models
rho = read_h5_model('../data/models/rho_with_salt_2D.h5')
epsilon = read_h5_model('../data/models/epsilon_with_salt_2D.h5')
delta = read_h5_model('../data/models/delta_with_salt_2D.h5')
theta = read_h5_model('../data/models/theta_with_salt_2D.h5')
phi = read_h5_model('../data/models/phi_with_salt_2D.h5')
m0 = read_h5_model('../data/models/migration_velocity_2D.h5')
dm = read_h5_model('../data/models/perturbation_2D.h5')

shape = (801, 267)
origin = (0.0, 0.0)
spacing = (12.5, 12.5)
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

# Source coordinates
nsrc = 2
rtm = np.zeros(model.shape, dtype='float32')
shots = []

for j in range(nsrc):
    print('Source number ', j)

    src_coords = np.linspace(start=500, stop=9500, num=nsrc)
    src_coordinates = [src_coords[j], 312.5]
    src = RickerSource(name='src', grid=model.grid, f0=.02, time_range=time, npoint=1)
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

    # Receivers
    nrec = 799
    rec_coords = np.empty((nrec, 2))
    rec_coords[:, 0] = np.linspace(12.5, 9987.5, nrec)
    rec_coords[:, 1] = 6.

    #########################################################################################

    # Devito operator
    tti = TTIPropagators(model, space_order=so)

    # Shot
    d_obs, u0, v0, summary1 = tti.born(src, rec_coords, save=True, sub=(12, 1))
    grad, summary2 = tti.gradient(d_obs, u0, v0, sub=(12, 1), isic=True)
    grad.data[:,0:66] = 0   # mute water column

    # Collect shots and stack images
    shots.append(resample(d_obs.data, t0, tn, time.num, nt))
    rtm += grad.data[model.nbpml:-model.nbpml, model.nbpml:-model.nbpml]

# Process summary
kernel, gflopss, gpointss, oi, ops = process_summaries([summary1, summary2])

# Plots
plt.figure(); plt.imshow(shots[0], vmin=-.1, vmax=.1, cmap='gray', aspect='auto')
plt.figure(); plt.imshow(np.transpose(rtm), vmin=-2e0, vmax=2e0, cmap='gray', aspect='auto')
plt.show()
