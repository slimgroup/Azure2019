# Imports
import sys, os
dir_path =  os.getcwd()[:23]
sys.path.insert(0, dir_path + '/src/AzureBatch/docker/tti_image/tti')

from AzureUtilities import segy_read
import matplotlib.pyplot as plt
import numpy as np


# First the shot record needs to be downloaded from Azure storage and saved in the `~/Azure2019/data/shots` directory. 
# The shot record can be downloaded via the web browser, the CLI or the Python SDK.
#

# Read shot record
shot_no = 1536
data, sourceX, sourceY, sourceZ, groupX, groupY, groupZ, tmax, dt, nt = segy_read(dir_path + '/data/shots/rec' + str(shot_no) + '.segy', ndims=3)
nrec_per_dim = 799
dx = 12.5
nshots = 3  # no. of shots to plot
tmax = (nt - 1)*dt / 1e3

#######################################################################################################################
# Common shot gathers

# Plot 'nshots' shot records closest to the source location
idx1 = np.abs(groupX - sourceX).argmin()
idx2 = np.where(groupY == dx)[0][np.abs(np.where(groupY == dx) - idx1).argmin()]
min_rec = idx2
max_rec = min_rec + nrec_per_dim*nshots

# Common shot record
data = data.reshape(-1, int(799*799))
plt.figure(figsize=(10,3));
plt.imshow(data[:, min_rec:max_rec], vmin=-.03, vmax=.03, cmap='seismic', aspect='auto', extent=(min_rec, max_rec, tmax, 0))
plt.xlabel('Trace no.'); plt.ylabel('Time [s]')
plt.tight_layout()

#######################################################################################################################
# Time slices

# Reshape data into cube
plt.figure(figsize=(10,3)); 
data = data.reshape(-1, nrec_per_dim, nrec_per_dim)

# Plot slice at times:
t1 = 2.0
t2 = 2.4
t3 = 2.8

plt.subplot(1,3,1); plt.imshow(data[int(t1*1e3/dt), :, :], vmin=-5e-2, vmax=5e-2, cmap='seismic')
plt.xticks(np.array([1,400,799]), ('1', '400', '799')); plt.yticks(np.array([1,400,799]), ('1', '400', '799'))
plt.xlabel('Inline Receiver No.'); plt.ylabel('Crossline Receiver No.')

plt.subplot(1,3,2); plt.imshow(data[int(t2*1e3/dt), :, :], vmin=-5e-2, vmax=5e-2, cmap='seismic')
plt.xticks(np.array([1,400,799]), ('1', '400', '799')); plt.yticks(np.array([1,400,799]), ('1', '400', '799'))
plt.xlabel('Inline Receiver No.'); plt.ylabel('Crossline Receiver No.')

plt.subplot(1,3,3); plt.imshow(data[int(t3*1e3/dt), :, :], vmin=-5e-2, vmax=5e-2, cmap='seismic')
plt.xticks(np.array([1,400,799]), ('1', '400', '799')); plt.yticks(np.array([1,400,799]), ('1', '400', '799'))
plt.xlabel('Inline Receiver No.'); plt.ylabel('Crossline Receiver No.')
plt.tight_layout()

plt.show()
