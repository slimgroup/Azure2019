# Scripts for running reverse-time migration

## Content

 - `generate_acquisition_geometry`: Create OBN acquisition geometry with dense source grid and random jittered receiver grid. Creates `hdf5` files for each source and receiver coordinates.

 - `overthrust_2D_local.py`: Test script for running 2D TTI RTM on a local desktop/laptop.

 - `overthrust_3D_local.py`: Script to locally verify the correct setup for 3D RTM. This script only reads the models, geometry, wavelet and sets up all operators, but does not actually compute anything.

 - `overthrust_3D_no_mpi.py`: Script for running 3D RTM using Azure Batch without MPI (i.e. one node per shot)

 - `overthrust_3D_limited_offset.py`: Script for running 3D RTM using Azure Batch with MPI. This is the script we used to generate the image in the Rice O&G conference abstract. To fit the forward wavefields in memory, we limit the maximum offset to 3.78 km in both X and Y direction.

 - `fetch_results.py`: Script for fetching images from Azure while the RTM script is running.


## Acquisition geometry

The source and receiver acquisition geometry as plotted below were generated with the script `generate_acquisition_geometry.py`. 

![](../documentation/source_grid.png)

![](../documentation/receiver_grid.png)
