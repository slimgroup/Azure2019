import numpy as np
import matplotlib.pyplot as plt
import h5py
from AzureUtilities import *
from scipy import interpolate, ndimage

def read_h5_model(filename):
    with h5py.File(filename, 'r') as f:
        filekey = list(f.keys())[0]
        data = np.array(f[filekey])
    return data

# Read models
vp = read_h5_model('vp_with_salt.h5')
vp_fine = read_h5_model('vp_fine_with_salt.h5')
rho = read_h5_model('rho_with_salt.h5')
epsilon = read_h5_model('epsilon_with_salt.h5')
delta = read_h5_model('delta_with_salt.h5')
theta = read_h5_model('theta_with_salt.h5')
phi = read_h5_model('phi_with_salt.h5')
m0 = read_h5_model('migration_velocity.h5')
dm = read_h5_model('perturbation.h5')

# Plots
plt.figure(); plt.imshow(np.transpose(vp_fine[400,:,:]), aspect='auto', cmap='jet', vmin=1.5, vmax=5.2); plt.title('Vp fine')
plt.figure(); plt.imshow(np.transpose(vp[400,:,:]), aspect='auto', cmap='jet', vmin=1.5, vmax=5.2); plt.title('Vp')
plt.figure(); plt.imshow(np.transpose(rho[400,:,:]), aspect='auto', cmap='jet'); plt.title('Rho')
plt.figure(); plt.imshow(np.transpose(delta[400,:,:]), aspect='auto', cmap='jet'); plt.title('Epsilon')
plt.figure(); plt.imshow(np.transpose(epsilon[400,:,:]), aspect='auto', cmap='jet'); plt.title('Delta')
plt.figure(); plt.imshow(np.transpose(theta[400,:,:]), aspect='auto', cmap='jet'); plt.title('Theta')
plt.figure(); plt.imshow(np.transpose(phi[400,:,:]), aspect='auto', cmap='jet'); plt.title('Phi')
plt.figure(); plt.imshow(np.transpose(np.sqrt(1/m0)[400,:,:]), aspect='auto', cmap='jet'); plt.title('Migration velocity')
plt.figure(); plt.imshow(np.transpose(dm[400,:,:]), aspect='auto', cmap='gray'); plt.title('dm')
plt.show()
