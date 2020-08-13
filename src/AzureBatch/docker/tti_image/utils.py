import numpy as np
import segyio as so
from scipy.signal import butter, sosfilt

import time

# Simple timer with message
def timer(start, message):
    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    info('{}: {:d}:{:02d}:{:02d}'.format(message, int(hours), int(minutes), int(seconds)))


# Time resampling for shot records
def resample(rec, num):
    if num == rec.time_range.num:
        return np.asarray(rec.data), np.asarray(rec.coordinates.data)

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


def save_rec_mpi(rec, src_coords, path, resample=None):
    comm = rec.grid.distributor.comm
    rank = comm.Get_rank()

    resample_num = resample or rec.time_range.num

    if rec.data.size != 0:
        rec_save, coords = resample(rec, resample_num)

        info("From rank %s, shot record of size %s, number of rec locations %s" % (rank, recx_save.shape, coords.shape))

        info("From rank %s, writing %s in recx, maximum value is %s" % (rank, recx_save.shape, np.max(recx_save)))
        segy_write(recx_save,
                   [src_coords[0]],
                   [src_coords[-1]],
                   coords[:, 0],
                   coords[:, -1],
                   2.0,  "%srecx_%s.segy" % (recloc, rank),
                   sourceY=[src_coords[1]],
                   groupY=coords[:, 1])
        info("From rank %s, writing %s in recy" % (rank, recy_save.shape))
        segy_write(recy_save,
                   [src_coords[0]],
                   [src_coords[-1]],
                   coords[:, 0],
                   coords[:, -1],
                   2.0,  "%srecy_%s.segy" % (recloc, rank),
                   sourceY=[src_coords[1]],
                   groupY=coords[:, 1])
        info("From rank %s, writing %s in recz" % (rank, recz_save.shape))
        segy_write(recz_save,
                   [src_coords[0]],
                   [src_coords[-1]],
                   coords[:, 0],
                   coords[:, -1],
                   2.0,  "%srecz_%s.segy" % (recloc, rank),
                   sourceY=[src_coords[1]],
                   groupY=coords[:, 1])



##############################################################################
# Filtering utils
def butter_highpass(highcut, fs, order=3):
    normal_cutoff = float(highcut) / fs
    sos = butter(order, normal_cutoff, analog=False, btype='highpass', output='sos')
    return sos

def butter_highpass_filter(data, highcut, fs, order=3):
    # Source: https://github.com/guillaume-chevalier/filtering-stft-and-laplace-transform
    sos = butter_highpass(highcut, fs, order=order)
    y = sosfilt(sos, data)
    return y

def butter_lowpass(lowcut, fs, order=3):
    normal_cutoff = float(lowcut) / fs
    sos = butter(order, normal_cutoff, analog=False, btype='lowpass', output='sos')
    return sos

def butter_lowpass_filter(data, lowcut, fs, order=3):
    # Source: https://github.com/guillaume-chevalier/filtering-stft-and-laplace-transform
    sos = butter_lowpass(lowcut, fs, order=order)
    y = sosfilt(sos, data)
    return y

def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    high = butter_highpass_filter(data, highcut, fs, order=order)
    return butter_lowpass_filter(high, lowcut, fs, order=order)