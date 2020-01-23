from tti_propagators import *
from model import *
from sources import *
import numpy as np
from devito import norm
from scipy import ndimage
import matplotlib.pyplot as plt
from AzureUtilities import read_h5_model
# Shot number (get from batch)
shot_no = 0
pathfile = '/app/model_phhil/'
# Read models
rho = read_h5_model(pathfile+'rho_with_salt_2D.h5')
epsilon = read_h5_model(pathfile+'epsilon_with_salt_2D.h5')
delta = read_h5_model(pathfile+'delta_with_salt_2D.h5')
theta = read_h5_model(pathfile+'theta_with_salt_2D.h5')
m0 = read_h5_model(pathfile+'migration_velocity_2D.h5')
dm = read_h5_model(pathfile+'perturbation_2D.h5')


n = (801, 267)
o = (0., 0.)
d = (12.5, 12.5)
so = 8
dt_full = .64

model = Model(shape=n, origin=o, spacing=d, vp=np.sqrt(1./m0), space_order=so,
              epsilon=epsilon, delta=delta, theta=theta, nbpml=40,
              dm=dm, rho=rho, dt=dt_full)

# Time axis
t0 = 0.
tn = 300.
dt = model.critical_dt
time = TimeAxis(start=t0, step=dt_full, stop=tn)
print(time.num)
src = RickerSource(name='src', grid=model.grid, f0=0.020, time_range=time, npoint=1)
src.coordinates.data[:, 0] = 5000.
src.coordinates.data[:, 1] = 350.

nrec = 501
rec_coords = np.empty((nrec, 2))
rec_coords[:, 0] = np.linspace(0., 10000., nrec)
rec_coords[:, 1] = 6.
####### RUN  #########
# Forward
op = TTIPropagators(model, space_order=so)

rec, u, v = op.forward(src, rec_coords, save=True)
grad = op.gradient(rec, u, v, isic=True)

linD, u, v, summary = op.born(src, rec_coords, sub=(4, 1), autotune=('aggressive', 'runtime'))
grad2 = op.gradient(linD, u, v, sub=(4, 1), isic=True, autotune=('basic', 'runtime'))

print(norm(grad))
print(norm(grad2))
print(norm(rec))
print(norm(linD))
