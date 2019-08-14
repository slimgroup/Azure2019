# Test blob IO functions
import numpy as np
from AzureUtilities import *
import matplotlib.pyplot as plt

# Random test array and model parameters
shape = (100, 120)
spacing = (6.25, 7.35)
origin = (13.34, 12.87)
X = np.random.randn(shape[0], shape[1]).astype('float32')

# Put/Get array
container = 'slim-bucket-common'
array_name = 'pwitte/models/test_array'
array_put(X, container, array_name)
X_rec = array_get(container, array_name)

print("Residual array: ", np.linalg.norm(X - X_rec))

# Put/Get model structure
container = 'slim-bucket-common'
model_name = 'pwitte/models/test_model'
model_put(X, origin, spacing, container, array_name)
X_rec, o_rec, s_rec = model_get(container, array_name)

print("Residual model: ", np.linalg.norm(X - X_rec))

# Get segy file
data_path = 'pwitte/data/'
filename = 'bp_observed_data_1005.segy'
data, sourceX, sourceZ, groupX, groupZ, tmax, dt, nt = segy_get(container, data_path, filename)

# Put segy file
segy_put(data, sourceX, sourceZ, groupX, groupZ, dt, container, data_path, filename, sourceY=None, groupY=None, elevScalar=-1000, coordScalar=-1000, keepFile=False)
