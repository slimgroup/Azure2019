import numpy as np
import segyio
import subprocess
import os
from scipy import interpolate
from devito import Eq, Operator
from azure.storage.blob import BlockBlobService, PublicAccess

blob_service = BlockBlobService(account_name='xxxx', account_key='xxxx')

####################################################################################################
# array put and get

def convert_to_string(t):
    if len(t) == 2:
        return str(t[0]) + 'S' + str(t[1])
    else:
        return str(t[0]) + 'S' + str(t[1]) + 'S' + str(t[2])

def convert_int_from_string(s):
    s_split = s.split('S')
    ndim = len(s_split)
    n1 = int(s_split[0])
    n2 = int(s_split[1])
    if ndim==2:
        n = (n1, n2)
    else:
        n3 = int(s_split[2])
        n = (n1, n2, n3)
    return n

def convert_float_from_string(s):
    s_split = s.split('S')
    ndim = len(s_split)
    d1 = float(s_split[0])
    d2 = float(s_split[1])
    if ndim==2:
        d = (d1, d2)
    else:
        d3 = float(s_split[2])
        d = (d1, d2, d3)
    return d

# write array
def array_put(blob, container, blob_name, index=0, count=None, validate_content=False):
    shape_str = convert_to_string(blob.shape)
    meta = {'dtype':str(blob.dtype), 'shape': shape_str}
    blob_service.create_blob_from_bytes(
        container,
        blob_name,
        blob.tostring(),   # blob
        index = index,    # start index in array of bytes
        count = count, # number of bytes to upload
        metadata = meta,  # Name-value pairs
        validate_content = validate_content
    )

# put array
def array_get(container, blob_name, start_range=None, end_range=None, validate_content=False):
    binary_blob = blob_service.get_blob_to_bytes(
        container,
        blob_name,
        start_range=start_range,
        end_range=end_range,
        validate_content=validate_content
    )
    meta = binary_blob.metadata
    shape = convert_int_from_string(meta['shape'])
    x = np.fromstring(binary_blob.content, dtype=meta['dtype'])
    return x.reshape(shape)

####################################################################################################
# model put and get

# write model
def model_put(model_blob, origin, spacing, container, blob_name, index=0, count=None, validate_content=False):
    shape_str = convert_to_string(model_blob.shape)
    origin_str = convert_to_string(origin)
    spacing_str = convert_to_string(spacing)
    meta = {'dtype':str(model_blob.dtype), 'shape': shape_str, 'origin': origin_str, 'spacing': spacing_str}
    blob_service.create_blob_from_bytes(
        container,
        blob_name,
        model_blob.tostring(),   # blob
        index = index,    # start index in array of bytes
        count = count, # number of bytes to upload
        metadata = meta,  # Name-value pairs
        validate_content = validate_content
    )

# read model
def model_get(container, blob_name, start_range=None, end_range=None, validate_content=False):
    binary_blob = blob_service.get_blob_to_bytes(
        container,
        blob_name,
        start_range=start_range,
        end_range=end_range,
        validate_content=validate_content
    )
    meta = binary_blob.metadata
    shape = convert_int_from_string(meta['shape'])
    origin = convert_float_from_string(meta['origin'])
    spacing = convert_float_from_string(meta['spacing'])
    x = np.fromstring(binary_blob.content, dtype=meta['dtype'])
    return x.reshape(shape), origin, spacing


####################################################################################################
# segy read

def segy_get(container, path, filename, ndims=2, keepFile=False):

    # copy from s3 to local volume
    subprocess.run(['az', 'storage', 'blob', 'download', '--container-name', container, '--name', path + filename,
    '--file', os.getcwd() + '/' + filename, '--output', 'table'])
    argout = segy_read(filename, ndims=ndims)

    if keepFile is False:
        subprocess.run(['rm', '-f', filename])

    return argout

def segy_read(filename, ndims=2):

    with segyio.open(filename, "r", ignore_geometry=True) as segyfile:
        segyfile.mmap()

        # Assume input data is for single common shot gather
        sourceX = segyfile.attributes(segyio.TraceField.SourceX)[0]
        sourceY = segyfile.attributes(segyio.TraceField.SourceY)[0]
        sourceZ = segyfile.attributes(segyio.TraceField.SourceSurfaceElevation)[0]
        groupX = segyfile.attributes(segyio.TraceField.GroupX)[:]
        groupY = segyfile.attributes(segyio.TraceField.GroupY)[:]
        groupZ = segyfile.attributes(segyio.TraceField.ReceiverGroupElevation)[:]
        dt = segyio.dt(segyfile)/1e3

        # Apply scaling
        elevScalar = segyfile.attributes(segyio.TraceField.ElevationScalar)[0]
        coordScalar = segyfile.attributes(segyio.TraceField.SourceGroupScalar)[0]

        if coordScalar < 0.:
            sourceX = sourceX / np.abs(coordScalar)
            sourceY = sourceY / np.abs(coordScalar)
            sourceZ = sourceZ / np.abs(elevScalar)
            groupX = groupX / np.abs(coordScalar)
            groupY = groupY / np.abs(coordScalar)
        elif coordScalar > 0.:
            sourceX = sourceX * np.abs(coordScalar)
            sourceY = sourceY * np.abs(coordScalar)
            sourceZ = sourceZ * np.abs(elevScalar)
            groupX = groupX * np.abs(coordScalar)
            groupY = groupY * np.abs(coordScalar)

        if elevScalar < 0.:
            groupZ = groupZ / np.abs(elevScalar)
        elif elevScalar > 0.:
            groupZ = groupZ * np.abs(elevScalar)

        nrec = len(groupX)
        nt = len(segyfile.trace[0])

        # Extract data
        data = np.zeros(shape=(nt, nrec), dtype='float32')
        for i in range(nrec):
            data[:,i] = segyfile.trace[i]
        tmax = (nt-1)*dt

    if ndims == 2:
        return data, sourceX, sourceZ, groupX, groupZ, tmax, dt, nt
    else:
        return data, sourceX, sourceY, sourceZ, groupX, groupY, groupZ, tmax, dt, nt


def segy_model_read(filename):

    with segyio.open(filename, "r", ignore_geometry=True) as segyfile:
        segyfile.mmap()

        # Assume input data is for single common shot gather
        sourceX = segyfile.attributes(segyio.TraceField.SourceX)
        dx = segyio.dt(segyfile)/1e3

        # Apply scaling
        coordScalar = segyfile.attributes(segyio.TraceField.SourceGroupScalar)[0]

        if coordScalar < 0.:
            sourceX = sourceX / np.abs(coordScalar)
        elif coordScalar > 0.:
            sourceX = sourceX * np.abs(coordScalar)

        nx = len(sourceX)
        nz = len(segyfile.trace[0])

        # Extract data
        data = np.zeros(shape=(nx, nz), dtype='float32')
        for i in range(nx):
            data[i,:] = segyfile.trace[i]

        return data, sourceX, dx


def segy_put(data, sourceX, sourceZ, groupX, groupZ, dt, container, path, filename, sourceY=None, groupY=None, elevScalar=-1000, coordScalar=-1000, keepFile=False):

    # Write segy file
    segy_write(data, sourceX, sourceZ, groupX, groupZ, dt, filename, sourceY=None, groupY=None, elevScalar=-1000, coordScalar=-1000)

    # copy from local volume to s3
    status = subprocess.run(['az', 'storage', 'blob', 'upload', '--container-name', container, '--file', filename, '--name', path+filename])

    if keepFile is False:
        subprocess.run(['rm', '-f', filename])

    return status


def segy_write(data, sourceX, sourceZ, groupX, groupZ, dt, filename, sourceY=None, groupY=None, elevScalar=-1000, coordScalar=-1000):

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



####################################################################################################
# Auxiliary modeling functions

# Add/subtract devito data w/ MPI
def add_rec(d1, d2):
    eq = Eq(d1, d1 + d2)
    op = Operator([eq])
    op()
    return d1

def sub_rec(d1, d2):
    eq = Eq(d1, d1 - d2)
    op = Operator([eq],subs={d2.indices[-1]: d1.indices[-1]})
    op()
    return d1

# Create 3D receiver grid from 1D x and y receiver vectors
def create_3D_grid(xrec, yrec, zrec):

    nxrec = len(xrec)
    nyrec = len(yrec)
    nrec_total = nxrec * nyrec

    rec = np.zeros(shape=(nrec_total, 3), dtype='float32')
    count = 0
    for j in range(nxrec):
        for k in range(nyrec):
            rec[count, 0] = xrec[j]
            rec[count, 1] = yrec[k]
            rec[count, 2] = zrec
            count += 1
    return rec


def restrict_model_to_receiver_grid(sx, gx, m, spacing, origin, sy=None, gy=None, buffer_size=500, numpy_coords=True):

    # Model parameters
    shape = m.shape
    ndim = len(shape)
    if ndim == 2:
        domain_size = ((shape[0] - 1) * spacing[0], (shape[1] - 1) * spacing[1])
    else:
        domain_size = ((shape[0] - 1) * spacing[0], (shape[1] - 1) * spacing[1], \
            (shape[2] - 1) * spacing[2])

    # Scan for minimum/maximum source/receiver coordinates
    min_x = np.min([np.min(sx), np.min(gx)])
    max_x = np.max([np.max(sx), np.max(gx)])
    if sy is not None and gy is not None:
        min_y = np.min([np.min(sy), np.min(gy)])
        max_y = np.max([np.max(sy), np.max(gy)])

    # Add buffer zone if possible
    min_x = np.max([origin[0], min_x - buffer_size])
    max_x = np.min([origin[0] + domain_size[0], max_x + buffer_size])
    #print("min_x: ", min_x)
    #print("max_x: ", max_x)
    if ndim == 3:
        min_y = np.max([origin[1], min_y - buffer_size])
        max_y = np.min([origin[1] + domain_size[1], max_y + buffer_size])
        #print("min_y: ", min_y)
        #print("max_y: ", max_y)

    # Extract model part
    nx_min = int(min_x / spacing[0])
    nx_max = int(max_x / spacing[0])
    #print("nx_min: ", nx_min)
    #print("nx_max: ", nx_max)
    ox = nx_min * spacing[0]
    oz = origin[-1]
    if ndim == 3:
        ny_min = int(min_y / spacing[1])
        ny_max = int(max_y / spacing[1])
        #print("ny_min: ", ny_min)
        #print("ny_max: ", ny_max)
        oy = ny_min * spacing[1]

    # Extract relevant part of model
    n_orig = shape
    #print("Original shape: ", n_orig)
    if ndim == 2:
        m = m[nx_min:nx_max+1, :]
        origin = (ox, oz)
    else:
        m = m[nx_min:nx_max+1, ny_min:ny_max+1, :]
        origin = (ox, oy, oz)
    shape = m.shape
    #print("New shape: ", shape)

    return m, shape, origin


def extent_gradient(shape_full, origin_full, shape_sub, origin_sub, spacing, g):

    nz = shape_full[-1]
    ndim = len(shape_full)

    nx_left = int((origin_sub[0] - origin_full[0]) / spacing[0])
    nx_right = shape_full[0] - shape_sub[0] - nx_left

    if ndim == 3:
        ny_left = int((origin_sub[1] - origin_full[1]) / spacing[1])
        ny_right = shape_full[1] - shape_sub[1] - ny_left

    if ndim == 2:
        block1 = np.zeros(shape=(nx_left, nz), dtype='float32')
        block2 = np.zeros(shape=(nx_right, nz), dtype='float32')
        g = np.concatenate((block1, g, block2), axis=0)
    else:
        block1 = np.zeros(shape=(nx_left, shape_sub[1], nz), dtype='float32')
        block2 = np.zeros(shape=(nx_right, shape_sub[1], nz), dtype='float32')
        g = np.concatenate((block1, g, block2), axis=0)
        del block1, block2
        block3 = np.zeros(shape=(shape_full[0], ny_left, nz), dtype='float32')
        block4 = np.zeros(shape=(shape_full[0], ny_right, nz), dtype='float32')
        g = np.concatenate((block3, g, block4), axis=1)

    return g


####################################################################################################
# Auxiliary AWS functions


def resample(data, t0, tn, nt_prev, nt_new):

    time_prev = np.linspace(start=t0, stop=tn, num=nt_prev)
    time_new = np.linspace(start=t0, stop=tn, num=nt_new)

    d_resamp = np.zeros(shape=(len(time_new), data.shape[1]), dtype='float32')
    for i in range(data.shape[1]):
        tck = interpolate.splrep(time_prev, data[:, i], k=3)
        d_resamp[:, i] = interpolate.splev(time_new, tck)
    return d_resamp


# Get chunk size of gradient
def get_chunk_size(g_size, num_chunks):

    average_size = int(g_size/num_chunks)
    num_residuals = g_size % num_chunks
    chunk_size = np.ones(num_chunks, dtype='int')*average_size
    if num_residuals > 0:
        for j in range(num_residuals):
            chunk_size[j] += 1
    return chunk_size
