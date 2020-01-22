import numpy as np
import matplotlib.pyplot as plt
import h5py
from AzureUtilities import *
from scipy import interpolate, ndimage

# Read model
vp_fine = np.load('../data/vp_with_salt_fine.npy').astype('float32')
vp = np.load('../data/vp_with_salt.npy').astype('float32')
delta = np.load('../data/delta_with_salt.npy').astype('float32')
epsilon = np.load('../data/epsilon_with_salt.npy').astype('float32')
theta = np.load('../data/theta_with_salt.npy').astype('float32')
phi = np.load('../data/phi_with_salt.npy').astype('float32')
rho = np.load('../data/rho_with_salt_fine.npy').astype('float32')

shape = (801, 801, 267)
spacing = (12.5, 12.5, 12.5)
origin = (0.0, 0.0, 0.0)
water_depth = 26    # gridpoints

vp_fine = vp_fine[:,:,0:-1:3]
with h5py.File('vp_fine_with_salt.h5', 'w') as data_file:
    data_file.create_dataset('vp_fine', data=vp_fine, dtype='float32')

vp = vp[:,:,0:-1:3]
vp[:,:,0:26] = vp[1,1,1]
with h5py.File('vp_with_salt.h5', 'w') as data_file:
    data_file.create_dataset('vp', data=vp, dtype='float32')

# Slowness
m_fine = (1.0 / vp_fine)**2
m = (1.0 / vp)**2
m0 = ndimage.gaussian_filter(m, sigma=(2,2,2))

# Migration velocity
m_mig = ndimage.gaussian_filter(m, sigma=(3,3,3))
m_mig[:,:,0:26] = m[:,:,0:26]
with h5py.File('migration_velocity.h5', 'w') as data_file:
    data_file.create_dataset('m0', data=m_mig, dtype='float32')

# Image
dm = m_fine - m0
with h5py.File('perturbation.h5', 'w') as data_file:
    data_file.create_dataset('dm', data=dm, dtype='float32')

rho = rho[:,:,0:-1:3]
rho = ndimage.gaussian_filter(rho, sigma=(2,2,2))
rho[:,:,0:26] = 1.0
with h5py.File('rho_with_salt.h5', 'w') as data_file:
    data_file.create_dataset('rho', data=rho, dtype='float32')

delta = delta[:,:,0:-1:3]
delta = ndimage.gaussian_filter(delta, sigma=(2,2,2))
delta[:,:,0:26] = 0.0
with h5py.File('delta_with_salt.h5', 'w') as data_file:
    data_file.create_dataset('delta', data=delta, dtype='float32')

epsilon = epsilon[:,:,0:-1:3]
epsilon = ndimage.gaussian_filter(epsilon, sigma=(2,2,2))
epsilon[:,:,0:26] = 0.0
with h5py.File('epsilon_with_salt.h5', 'w') as data_file:
    data_file.create_dataset('epsilon', data=epsilon, dtype='float32')

theta = theta[:,:,0:-1:3]
theta = ndimage.gaussian_filter(theta, sigma=(2,2,2))
theta[:,:,0:26] = 0.0
with h5py.File('theta_with_salt.h5', 'w') as data_file:
    data_file.create_dataset('theta', data=theta, dtype='float32')

phi = phi[:,:,0:-1:3]
phi = ndimage.gaussian_filter(phi, sigma=(2,2,2))
phi[:,:,0:26] = 0.0
with h5py.File('phi_with_salt.h5', 'w') as data_file:
    data_file.create_dataset('phi', data=phi, dtype='float32')

# Extract 2D slices
vp2D = vp[:, 400,:]
vp_fine2D = vp_fine[:, 400,:]
rho2D = rho[:,400,:]
epsilon2D = epsilon[:,400,:]
delta2D = delta[:,400,:]
theta2D = theta[:,400,:]
phi2D = phi[:,400,:]
m0_2D = m0[:,400,:]
dm2D = dm[:,400,:]

# Save 2D slices
with h5py.File('migration_velocity_2D.h5', 'w') as data_file:
    data_file.create_dataset('m0', data=m0_2D, dtype='float32')

with h5py.File('perturbation_2D.h5', 'w') as data_file:
    data_file.create_dataset('dm', data=dm2D, dtype='float32')

with h5py.File('rho_with_salt_2D.h5', 'w') as data_file:
    data_file.create_dataset('rho', data=rho2D, dtype='float32')

with h5py.File('delta_with_salt_2D.h5', 'w') as data_file:
    data_file.create_dataset('delta', data=delta2D, dtype='float32')

with h5py.File('epsilon_with_salt_2D.h5', 'w') as data_file:
    data_file.create_dataset('epsilon', data=epsilon2D, dtype='float32')

with h5py.File('theta_with_salt_2D.h5', 'w') as data_file:
    data_file.create_dataset('theta', data=theta2D, dtype='float32')

with h5py.File('phi_with_salt_2D.h5', 'w') as data_file:
    data_file.create_dataset('phi', data=phi2D, dtype='float32')

# Plots
plt.figure(); plt.imshow(np.transpose(np.sqrt(1/m0)[400,:,:]), aspect='auto', cmap='jet', vmin=1.5, vmax=5.2); plt.title('m0')
plt.figure(); plt.imshow(np.transpose(np.sqrt(1/m_mig)[400,:,:]), aspect='auto', cmap='jet', vmin=1.5, vmax=5.2); plt.title('m_mig')
plt.figure(); plt.imshow(np.transpose(dm[400,:,:]), aspect='auto', cmap='gray', vmin=-0.1, vmax=0.1); plt.title('dm')
plt.figure(); plt.imshow(np.transpose(rho[400,:,:]), aspect='auto', cmap='jet'); plt.title('Rho')
plt.figure(); plt.imshow(np.transpose(delta[400,:,:]), aspect='auto', cmap='jet'); plt.title('Epsilon')
plt.figure(); plt.imshow(np.transpose(epsilon[400,:,:]), aspect='auto', cmap='jet'); plt.title('Delta')
plt.figure(); plt.imshow(np.transpose(theta[400,:,:]), aspect='auto', cmap='jet'); plt.title('Theta')
plt.figure(); plt.imshow(np.transpose(phi[400,:,:]), aspect='auto', cmap='jet'); plt.title('Phi')
plt.show()
