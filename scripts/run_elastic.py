# Azure imports
# Azure imports
from argparse import ArgumentParser

import numpy as np
from sympy import sin, Abs
import segyio
from scipy import interpolate

# Devito imports
from devito import *
from devito.logger import info
from devito.symbolics.extended_sympy import eval_bhaskara_sin        
 
from examples.seismic import demo_model, RickerSource, Receiver, TimeAxis, ModelElastic
import time


def timer(start, message):
    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    info('{}: {:d}:{:02d}:{:02d}'.format(message, int(hours), int(minutes), int(seconds)))

#######################################################################################
# Time resampling for shot records
def resample(rec, num):
    start, stop = rec._time_range.start, rec._time_range.stop
    dt0 = rec._time_range.step

    new_time_range = TimeAxis(start=start, stop=stop, num=num)
    dt = new_time_range.step

    to_interp = np.asarray(rec.data)
    data = np.zeros((num, to_interp.shape[1]))

    for i in range(to_interp.shape[1]):
        tck = interpolate.splrep(rec._time_range.time_values, to_interp[:, i], k=3)
        data[:, i] = interpolate.splev(new_time_range.time_values, tck)

    coords_loc = np.asarray(rec.coordinates.data)
    # Return new object
    return data, coords_loc

#######################################################################################
# Segy writer for shot records
def segy_write(data, sourceX, sourceZ, groupX, groupZ, dt, filename, sourceY=None,
               groupY=None, elevScalar=-1000, coordScalar=-1000):

    nt = data.shape[0]
    nsrc = 1
    nxrec = len(groupX)
    if sourceY is None and groupY is None:
        sourceY = np.zeros(1, dtype='int')
        groupY = np.zeros(nxrec, dtype='int')
    nyrec = len(groupY)

    # Create spec object
    spec = segyio.spec()
    spec.ilines = np.arange(nxrec)    # dummy trace count
    spec.xlines = np.zeros(1, dtype='int')  # assume coordinates are already vectorized for 3D
    spec.samples = range(nt)
    spec.format=1
    spec.sorting=1

    with segyio.create(filename, spec) as segyfile:
        for i in range(nxrec):
            segyfile.header[i] = {
                segyio.su.tracl : i+1,
                segyio.su.tracr : i+1,
                segyio.su.fldr : 1,
                segyio.su.tracf : i+1,
                segyio.su.sx : int(np.round(sourceX[0] * np.abs(coordScalar))),
                segyio.su.sy : int(np.round(sourceY[0] * np.abs(coordScalar))),
                segyio.su.selev: int(np.round(sourceZ[0] * np.abs(elevScalar))),
                segyio.su.gx : int(np.round(groupX[i] * np.abs(coordScalar))),
                segyio.su.gy : int(np.round(groupY[i] * np.abs(coordScalar))),
                segyio.su.gelev : int(np.round(groupZ[i] * np.abs(elevScalar))),
                segyio.su.dt : int(dt*1e3),
                segyio.su.scalel : int(elevScalar),
                segyio.su.scalco : int(coordScalar)
            }
            segyfile.trace[i] = data[:, i]
        segyfile.dt=int(dt*1e3)

def save_rec(recx, recy, recz, src_coords, recloc, nt, dt):
    comm = recx.grid.distributor.comm
    rank = comm.Get_rank()

    if recx.data.size != 0:
        recx_save, coords = resample(recx, nt)
        recy_save, _ = resample(recy, nt)
        recz_save, _ = resample(recz, nt)

        info("From rank %s, shot record of size %s, number of rec locations %s" % (rank, recx_save.shape, coords.shape))

        info("From rank %s, writing %s in recx, maximum value is %s" % (rank, recx_save.shape, np.max(recx_save)))
        segy_write(recx_save,
                   [src_coords[0]],
                   [src_coords[-1]],
                   coords[:, 0],
                   coords[:, -1],
                   dt,  "%srecx_%s.segy" % (recloc, rank),
                   sourceY=[src_coords[1]],
                   groupY=coords[:, 1])
        info("From rank %s, writing %s in recy" % (rank, recy_save.shape))
        segy_write(recy_save,
                   [src_coords[0]],
                   [src_coords[-1]],
                   coords[:, 0],
                   coords[:, -1],
                   dt,  "%srecy_%s.segy" % (recloc, rank),
                   sourceY=[src_coords[1]],
                   groupY=coords[:, 1])
        info("From rank %s, writing %s in recz" % (rank, recz_save.shape))
        segy_write(recz_save,
                   [src_coords[0]],
                   [src_coords[-1]],
                   coords[:, 0],
                   coords[:, -1],
                   dt,  "%srecz_%s.segy" % (recloc, rank),
                   sourceY=[src_coords[1]],
                   groupY=coords[:, 1])

#######################################################################################
def scale_vs(in_slice):
    """
    Scale S wave velocities in 100ms/s-800ms/ to 600ms.-800m/s as in the seam setup in the
    doc with a linear scale ie f(100) = 600, f(800) = 800 and f(x) =
    """
    in_slice[(in_slice <= .8) & (in_slice > 0)] = 2.0/7.0*(in_slice[(in_slice <= .8) & (in_slice > 0)] - .1) + .6
    return in_slice

#######################################################################################

t00 = time.time()
t0 = time.time()
####### Filter arguments
description = ("Example script for elastic operators.")
parser = ArgumentParser(description=description)
parser.add_argument("--id",  dest='shot_id', default=1, type=int,
                    help="Shot number")
parser.add_argument("--recloc",  dest='recloc', default="", type=str,
                    help="Path to results directory in blob")
parser.add_argument("--uloc",  dest='uloc', default="", type=str,
                    help="Path to wavefield directory in blob")
parser.add_argument("--modelloc",  dest='modelloc', default="", type=str,
                    help="Path to results directory in blob")
args = parser.parse_args()

# Get inputs
shot_id = args.shot_id
recloc = args.recloc
uloc = args.uloc
modelloc = args.modelloc


configuration['log-level'] = 'DEBUG'
# Some parameters
so = 8
nbpml = 40

timer(t0, 'Args process')
t0 = time.time()
#######################################################################################
# Grid is 3501x4001x1500 ith 40 abc points on each side
shape_domain = (1831, 2081, 1581)
grid = Grid(shape_domain, extent=(36600., 41600., 16600.), origin=(-800., -800., -800.))

# Elastic parameters
lam = Function(name="lam", grid=grid, space_order=0, is_parameter=True)
mu = Function(name="mu", grid=grid, space_order=0, is_parameter=True)
irho = Function(name="irho", grid=grid, space_order=0, is_parameter=True)

# Absorbing mask
damp = Function(name="damp", grid=grid, space_order=0, is_parameter=True)

# Stress and particle velocities
v = VectorTimeFunction(name="v", grid=grid, space_order=so, time_order=1)
tau = TensorTimeFunction(name="tau", grid=grid, space_order=so, time_order=1)

# symbol for dt
s = grid.time_dim.spacing
dt_cfl = 1.


timer(t0, 'Create Functions')
t0 = time.time()
#######################################################################################
# Get MPI info
comm = grid.distributor.comm
rank =  comm.Get_rank()
size =  comm.size
#######################################################################################
# Initialize model params
vp = np.ones(shape_domain) * 1.5
vp[:, :, 500:] = 2.5
vp[:, :, 1000:] = 3.5
vp[:, :, 1200:] = 4.5

# First aprrox is vs = (vp - 1.5)/1.5 and rho = vp/2 + .25
lam.data[:] = (vp/2 + .25) * vp**2
mu.data[:] = (vp/2 + .25) * (vp**2 - ((vp - 1.5)/1.5)**2)
irho.data[:] = 1 / (vp/2 + .25)

# delete lare array not needed anymore.
vp = 0

timer(t0, 'Read segy models')
t0 = time.time()
#######################################################################################
# Initialize damp mask
dampcoeff = 1.5 * np.log(1.0 / 0.001) / (nbpml)
eqs = [Eq(damp, 1.0)]
for d in damp.dimensions:
    # left
    dim_l = SubDimension.left(name='abc_%s_l'%d.name, parent=d, thickness=nbpml)
    pos = Abs((nbpml - (dim_l - d.symbolic_min) + 1) / float(nbpml))
    val = -dampcoeff * (pos - eval_bhaskara_sin(2*np.pi*pos)/(2*np.pi))
    eqs +=  [Inc(damp.subs({d: dim_l}), val/d.spacing)]
    # right
    dim_r = SubDimension.right(name='abc_%s_r'%d.name, parent=d, thickness=nbpml)
    pos = Abs((nbpml - (d.symbolic_max - dim_r) + 1) / float(nbpml))
    val = -dampcoeff * (pos - eval_bhaskara_sin(2*np.pi*pos)/(2*np.pi))
    eqs +=  [Inc(damp.subs({d: dim_r}), val/d.spacing)]

Operator(eqs)()
# damp.data.fill(0.)


timer(t0, 'Setup damp')
t0 = time.time()
#######################################################################################
# Source
# Source is at the middle of the domain at 10m depth
tstart, tn = 0., 16000.
time_range = TimeAxis(start=tstart, stop=tn, step=dt_cfl)

f0 = 0.010 
r = np.pi * f0 * (time_range.time_values - 1./f0)
wavelet = (1 - 2. * r**2) * np.exp(-r**2)


src_coords = np.array([17500., 20000., 10.])
src = RickerSource(name='src', grid=grid, f0=0.010, time_range=time_range)
src.coordinates.data[0, :] = src_coords

src_P = src.inject(field=tau.forward[0, 0], expr=s*src)
src_P +=src.inject(field=tau.forward[1, 1], expr=s*src)
src_P += src.inject(field=tau.forward[2, 2], expr=s*src)
from scipy.signal import butter, sosfilt

def butter_lowpass(cutoff, nyq_freq, order=5):
    normal_cutoff = float(cutoff) / nyq_freq
    sos = butter(order, normal_cutoff, analog=False, btype='lowpass', output='sos')
    return sos

def butter_lowpass_filter(data, cutoff_freq, f_sample, order=3):
    sos = butter_lowpass(cutoff_freq, f_sample, order=order)
    y = sosfilt(sos, data)
    return y
    
src.data[:, 0] = butter_lowpass_filter(wavelet, 0.020, 1./dt_cfl)

#######################################################################################
# The receiver
# Rec, OBC every 100m in both directions, measures particle velocities vx, by and vz separately
nrecx = 351
nrecy = 401
nrec = nrecx * nrecy
rec_coordinates = np.empty((nrecx, nrecy, 3))
for i in range(nrecx):
    for j in range(nrecy):
        rec_coordinates[i, j, 0] = i * 100.
        rec_coordinates[i, j, 1] = j * 100.
        rec_coordinates[i, j, 2] = wb[i, j]*10.

rec_coordinates = np.reshape(rec_coordinates, (nrec, 3))

recx = Receiver(name="recx", grid=grid, npoint=nrec, time_range=time_range)
recx.coordinates.data[:, :] = rec_coordinates[:, :]

recy = Receiver(name="recy", grid=grid, npoint=nrec, time_range=time_range)
recy.coordinates.data[:, :] = rec_coordinates[:, :]

recz = Receiver(name="recz", grid=grid, npoint=nrec, time_range=time_range)
recz.coordinates.data[:, :] = rec_coordinates[:, :]

rec_term = recx.interpolate(expr=v[0])
rec_term += recy.interpolate(expr=v[1])
rec_term += recz.interpolate(expr=v[2])


timer(t0, 'Setup geometry')
t0 = time.time()

#######################################################################################
# Operator
# Now let's try and create the staggered updates
# fdelmodc reference implementation
u_v = Eq(v.forward, damp * (v + s*irho*div(tau)))
u_t = Eq(tau.forward, damp *  (tau + s * (lam * diag(div(v.forward)) +
                                          mu * (grad(v.forward) + grad(v.forward).T))))

subs = {d.spacing: s for d, s in zip(grid.dimensions, grid.spacing)}
subs.update({s: dt_cfl})

op = Operator([u_v] + [u_t] + src_P + rec_term, subs=subs, name="ElasticForward")

timer(t0, 'Setup Operator')
t0 = time.time()
#######################################################################################

op.cfunction

timer(t0, 'Process + compile operator')
t0 = time.time()

info("From rank %s out of %s, Running forward modeling" % (rank, size))
# Run
op(autotune=('aggressive', 'runtime'))

timer(t0, 'Ran elastic forward modeling for 5sec recording')
t0 = time.time()

#######################################################################################
# Check output

info("Nan values : (%s, %s)" % (np.any(np.isnan(v[0].data)), np.any(np.isnan(recx.data))))
info("saving shot records")
save_rec(recx, recy, recz, src_coords, recloc)
timer(t0, 'Saved data')
t0 = time.time()
#######################################################################################
# Save wavefields for plotting

np.save("%svx_%s.npy" % (uloc, rank), np.asarray(v[0].data[0]))
np.save("%svy_%.npy" % (uloc, rank), np.asarray(v[1].data[0]))
np.save("%svz_%r,=.npy" % (uloc, rank), np.asarray(v[2].data[0]))

np.save("%stxx_%s.npy" % (uloc, rank), np.asarray(tau[0, 0].data[0]))
np.save("%styy_%s.npy" % (uloc, rank), np.asarray(tau[1, 1].data[0]))
np.save("%stzz_%s.npy" % (uloc, rank), np.asarray(tau[2, 2].data[0]))
np.save("%stxy_%s.npy" % (uloc, rank), np.asarray(tau[0, 1].data[0]))
np.save("%stxz_%s.npy" % (uloc, rank), np.asarray(tau[0, 2].data[0]))
np.save("%styz_%s.npy" % (uloc, rank), np.asarray(tau[1, 2].data[0]))

timer(t0, 'Saved wavefields')
t0 = time.time()
#######################################################################################
# Run rest of the simulation
op(autotune=('aggressive', 'runtime'), time_m=5000)

timer(t0, 'Ran elastic forward modeling for 11 extra sec of recording')
t0 = time.time()

#######################################################################################
# Check output

info("Nan values : (%s, %s)" % (np.any(np.isnan(v[0].data)), np.any(np.isnan(recx.data))))
info("saving shot records")
save_rec(recx, recy, recz, src_coords, recloc)
timer(t0, 'Saved data')
t0 = time.time()

timer(t00, 'Total runtime')
