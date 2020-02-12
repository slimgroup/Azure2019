from tti_propagators import *
from model import *
from sources import *
import numpy as np
from devito import norm
from scipy import ndimage, interpolate
import matplotlib.pyplot as plt
from AzureUtilities import read_h5_model
from utils import butter_bandpass_filter
# Shot number (get from batch)
shot_no = 0
pathfile = '/data/mlouboutin3/overthrust/model_phhil/'


##########################################################################################

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



# Read models
rho = read_h5_model(pathfile+'rho_with_salt.h5')[::4, ::4, :]
epsilon = read_h5_model(pathfile+'epsilon_with_salt.h5')[::4, ::4, :]
delta = read_h5_model(pathfile+'delta_with_salt.h5')[::4, ::4, :]
theta = read_h5_model(pathfile+'theta_with_salt.h5')[::4, ::4, :]
phi = read_h5_model(pathfile+'phi_with_salt.h5')[::4, ::4, :]
vp = read_h5_model(pathfile+'vp_fine_with_salt.h5')[::4, ::4, :]


n = (201, 201, 267)
o = (0., 0., 0.)
d = (12.5, 12.5, 12.5)
so = 4

model = Model(shape=n, origin=o, spacing=d, vp=vp, space_order=so,
              epsilon=epsilon, delta=delta, theta=theta, phi=phi, nbpml=40,
              rho=rho)

# Time axis
tstart = 0.
tn = 500.
dt = model.critical_dt
time_axis = TimeAxis(start=tstart, step=dt, stop=tn)
#########################################################################################
# Source and receiver geometries
nrecx = 201
nrecy = 201
nrec = nrecx * nrecy

rec_coordinates = np.empty((nrecx, nrecy, 3))
for i in range(nrecx):
    for j in range(nrecy):
        rec_coordinates[i, j, 0] = (i + 1) * d[0] * 4
        rec_coordinates[i, j, 1] = (j +1) * d[1] * 4
        rec_coordinates[i, j, 2] = 6
rec_coordinates = np.reshape(rec_coordinates, (nrec, 3))


src_coords = np.empty((1, len(d)))
src_coords[0, :] = np.array(model.domain_size) * .5
src_coords[0, -1] = 287.5
src = RickerSource(name='src', grid=model.grid, f0=0.020, time_range=time_axis, npoint=1)
src.coordinates.data[0, :] = src_coords[:]

wavelet = np.concatenate((np.load("/data/mlouboutin3/overthrust/data/wavelet/wavelet.npy"), np.zeros((100,))))
twave = [i*1.2 for i in range(wavelet.shape[0])]
tnew = [i*dt for i in range(int(1 + (twave[-1]-tstart) / dt))]
fq = interpolate.interp1d(twave, wavelet, kind='linear')
q_custom = np.zeros((time_axis.num, 1))
q_custom[:len(tnew), 0] = fq(tnew)
q_custom[:, 0] = butter_bandpass_filter(q_custom[:, 0], .005, .030, 1/dt)
q_custom[:, 0] = 1e1 * q_custom[:, 0] / np.max(q_custom[:, 0])

src.data[:, 0] = q_custom[:, 0]

####### RUN  #########
# Forward
op = TTIPropagators(model, space_order=so)

rec, u, v = op.forward(src, rec_coordinates, autotune=('aggressive', 'runtime'))
from IPython import embed; embed()
# grad = op.gradient(rec, u, v, isic=True)
#
# linD, u, v, summary = op.born(src, rec_coords, sub=(4, 1), autotune=('aggressive', 'runtime'))
# grad2 = op.gradient(linD, u, v, sub=(4, 1), isic=True, autotune=('basic', 'runtime'))
#
# print(norm(grad))
# print(norm(grad2))
# print(norm(rec))
# print(norm(linD))
