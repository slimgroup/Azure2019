# Create source and receiver grid

import numpy as np
import h5py
import matplotlib.pyplot as plt

# Model domain
x_max = 10000   # lateral extension in x [m]
y_max = 10000   # lateral extension in y [m]

###################################################################################################
# Source grid

# Source grid spacing
dsx = 12.5  # in m
dsy = 12.5  # in m

# Minimum source position
sx_min = dsx
sy_min = dsy

# Maximum source position
sx_max = x_max - dsx
sy_max = y_max - dsy

# Source depth
zsrc = 6.0

# Coordinate ranges
xrange_src = np.arange(start=sx_min, stop=sx_max + dsx, step=dsx)
yrange_src = np.arange(start=sy_min, stop=sy_max + dsy, step=dsy)

# No. of sources
nx_rec = len(xrange_src)
ny_rec = len(yrange_src)

Y_src, X_src = np.meshgrid(yrange_src, xrange_src)
I = np.ones(X_src.shape)
Z_src = I*zsrc

# Coordinates [X, Y, Z, I] (I: off the grid identifier, 1 if on the grid, 0 else)
src_coordinates = np.concatenate((X_src.reshape(-1,1), Y_src.reshape(-1,1), 
    Z_src.reshape(-1,1), I.reshape(-1,1)), axis=1)


###################################################################################################
# Jittered receiver grid

def generate_jittered_indices(ns, ds, rndfactor, p, rseed, boatspeed, tfireint_min, tdelay):
#   p=4 --> 75 % subsampling, ds = 12.5m
#   p=2 --> 50 % subsampling, ds = 25 m
    np.random.seed(rseed)
    dtfirejitb1arr1 = tfireint_min + np.random.rand(1, int(ns/p))*(2*tfireint_min)
    tfirejitb1arr1 = np.cumsum(dtfirejitb1arr1)
    tfirejitb1arr1 = tfirejitb1arr1 - tdelay
    tfirejitb1arr1 = np.round(rndfactor[0]*tfirejitb1arr1)/rndfactor[0]
    sjitb1arr1 = np.round(rndfactor[1]*boatspeed*tfirejitb1arr1)/rndfactor[1]
    return (sjitb1arr1/ds).astype(int)


# Underlying dense receiver grid spacing
drx = 50.0  # in m
dry = 50.0  # in m

# Minimum receiver position
rx_min = drx
ry_min = dry

# Maximum receiver position
rx_max = x_max - drx
ry_max = y_max - dry

# OBN receiver depth
zsrc = 740.0

# Coordinate ranges
xrange_rec = np.arange(start=rx_min, stop=rx_max + drx, step=drx)
yrange_rec = np.arange(start=ry_min, stop=ry_max + dry, step=dry)

nx_rec = len(xrange_rec)
ny_rec = len(yrange_rec)
n_rec = nx_rec * ny_rec

# Underlying dense receiver grid    [m]
Y_rec, X_rec = np.meshgrid(yrange_rec, xrange_rec)

# Jittered grid
fac = 8
p = 2*fac
ds = 25/fac
rndfactor = np.array([1000, 100])
rseed = 3402
boatspeed = 2.5
tfireint_min = 10
tdelay = 10
n_rec_jit = int(n_rec/p)

# Jittered grid
indices = generate_jittered_indices(n_rec, ds, rndfactor, p, rseed, boatspeed, tfireint_min, tdelay)
X_idx_jit, Y_idx_jit = np.unravel_index(indices, (nx_rec, ny_rec))

X_rec_jit = X_idx_jit*drx + drx
Y_rec_jit = Y_idx_jit*dry + dry
Z_rec_jit = np.ones(n_rec_jit)*zsrc

# Random dither between -5 m and 5 m
dith_x = (-5 + (5-(-5))*np.random.rand(n_rec_jit, 1))
dith_y = (-5 + (5-(-5))*np.random.rand(n_rec_jit, 1))

X_rec_dith = X_rec_jit.reshape(-1,1) + dith_x
Y_rec_dith = Y_rec_jit.reshape(-1,1) + dith_y

# Off the grid identifier: 1 --> On the grid; 0 --> Off the grid
I_jit = np.ones(n_rec_jit)
I_dith = np.zeros(n_rec_jit)

for i in range(n_rec_jit):
    if X_rec_dith[i] - X_rec_jit[i] <= 1e-9 and Y_rec_dith[i] - Y_rec_jit[i] <= 1e-9:
        I_dith[i] = 1

# Gather all coordinates
rec_coordinates_jittered = np.concatenate((X_rec_jit.reshape(-1,1), Y_rec_jit.reshape(-1,1), 
    Z_rec_jit.reshape(-1,1), I_jit.reshape(-1,1)), axis=1)

rec_coordinates_dithered = np.concatenate((X_rec_dith.reshape(-1,1), Y_rec_dith.reshape(-1,1), 
    Z_rec_jit.reshape(-1,1), I_dith.reshape(-1,1)), axis=1)

# Save sources as receiver coordinates and vice versa (due to source-receiver reciprocity)
with h5py.File('src_coordinates_new.h5', 'w') as data_file:
    data_file.create_dataset('xsrc', data=rec_coordinates_dithered)

with h5py.File('rec_coordinates_new.h5', 'w') as data_file:
    data_file.create_dataset('rec', data=src_coordinates)

# Plot grids
plt.figure(); plt.plot(rec_coordinates_dithered[:,0], rec_coordinates_dithered[:,1], 'o')
plt.show()