# Base imports
import sys, os
sys.path.insert(0, '/app/tti')
from argparse import ArgumentParser
import numpy as np
import time

# Devito imports
from devito.logger import info  

# tti imports from docker image
from model import *
from sources import *
from tti_propagators import *

# segy
import segyio as so

# Interpolation and filtering utils
from scipy import interpolate
from utils import butter_bandpass_filter, butter_lowpass_filter

# Azure utilities
from AzureUtilities import read_h5_model, write_h5_model, process_summaries, read_coordinates, save_rec

def timer(start, message):
    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    info('{}: {:d}:{:02d}:{:02d}'.format(message, int(hours), int(minutes), int(seconds)))


#######################################################################################

t0 = time.time()

####### Filter arguments
description = ("3D modeling on tti overdone")
parser = ArgumentParser(description=description)
parser.add_argument("--id",  dest='shot_id', default=1, type=int,
                    help="Shot number")
parser.add_argument("--recloc",  dest='recloc', default="", type=str,
                    help="Path to results directory in blob")
parser.add_argument("--modelloc",  dest='modelloc', default="", type=str,
                    help="Path to model directory in blob")
parser.add_argument("--geomloc",  dest='geomloc', default="", type=str,
                    help="Path to geometry directory in blob")
parser.add_argument("--fs",  dest='freesurf', default=False, action='store_true',
                    help="Freesurface")
args = parser.parse_args()

# Get inputs
shot_id = args.shot_id
recloc = args.recloc
modelloc = args.modelloc
geomloc = args.geomloc
freesurf = args.freesurf
# Some parameters
space_order = 12
nbpml = 40
timer(t0, 'Args process')
t0 = time.time()

#########################################################################################

# Read models
rho = read_h5_model(modelloc + 'rho_with_salt.h5')
epsilon = read_h5_model(modelloc + 'epsilon_with_salt.h5')
delta = read_h5_model(modelloc + 'delta_with_salt.h5')
theta = read_h5_model(modelloc + 'theta_with_salt.h5')
phi = read_h5_model(modelloc + 'phi_with_salt.h5')
vp = read_h5_model(modelloc + 'vp_fine_with_salt.h5')
shape = (801, 801, 267)
origin = (0.0, 0.0, 0.0)
spacing = (12.5, 12.5, 12.5)
model = Model(shape=shape, origin=origin, spacing=spacing, vp=vp, space_order=space_order,
              epsilon=epsilon, delta=delta, theta=theta, phi=phi, rho=rho, nbpml=nbpml)
#######################################################################################

# Get MPI info
comm = model.grid.distributor.comm
rank =  comm.Get_rank()
size =  comm.size
info("Min value in vp is %s " % (np.min(model.vp.data[:])))
timer(t0, 'Read segy models')
t0 = time.time()

#########################################################################################
# Model a 2D shot

# Time axis
tstart = 0.
tn = 500.
dt = model.critical_dt
f0 = 0.025
time_axis = TimeAxis(start=tstart, step=dt, stop=tn)

#########################################################################################

# Source and receiver geometries
nrecx = 101
nrecy = 401
nrec= nrecx * nrecy

src_coords = np.empty((1, len(spacing)))
src_coords[0, :] = np.array(model.domain_size) * .5
src_coords[0, -1] = 287.5

rec_coordinates = np.empty((nrecx, nrecy, 3))
for i in range(nrecx):
    for j in range(nrecy):
        rec_coordinates[i, j, 0] = (i + 1) * spacing[0]
        rec_coordinates[i, j, 1] = (j + 1) * spacing[1]
        rec_coordinates[i, j, 2] = 6.0

rec_coordinates = np.reshape(rec_coordinates, (nrec, 3))


src = RickerSource(name='src', grid=model.grid, f0=f0, time_range=time_axis, npoint=1)
src.coordinates.data[0, :] = src_coords[:]

wavelet = np.concatenate((np.load("%swavelet.npy"%geomloc), np.zeros((100,))))
twave = [i*1.2 for i in range(wavelet.shape[0])]
tnew = [i*dt for i in range(int(1 + (twave[-1]-tstart) / dt))]
fq = interpolate.interp1d(twave, wavelet, kind='linear')
q_custom = np.zeros((time_axis.num, 1))
q_custom[:len(tnew), 0] = fq(tnew)
q_custom[:, 0] = butter_bandpass_filter(q_custom[:, 0], .005, .030, 1/dt)
q_custom[:, 0] = 1e1 * q_custom[:, 0] / np.max(q_custom[:, 0])
src.data[:, 0] = q_custom[:, 0]

timer(t0, 'Setup geometry')
t0 = time.time()


#########################################################################################

# Propagator
tti = TTIPropagators(model, space_order=space_order)

#########################################################################################

# Data
info("Starting forward modeling")
d_obs, u, v = tti.forward(src, rec_coordinates, autotune=('aggressive', 'runtime'))
timer(t0, 'Run forward')
t0 = time.time()

#######################################################################################

# Check output
info("Nan values : %s" % np.any(np.isnan(d_obs.data[:])))
info("saving shot records %srecloc%s%s" % (recloc, rank, shot_id))
nt_new = 126

# Save shot
filename = recloc + 'overthrust_3D_shot_no_' + str(shot_id)
save_rec(d_obs, src_coords.reshape(3), filename, nt_new)
timer(t0, 'Saved data')
