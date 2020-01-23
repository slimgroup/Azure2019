# New generation seismic processing

An event-driven workflow for severless seismic imaging on Azure.

## Prerequisites

- [Docker](https://www.docker.com/products/docker-desktop)

- [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest)

- [Batch Shipyard](https://github.com/Azure/batch-shipyard)

- For the event-driven image summation: [Azure Functions Core Tools](https://docs.microsoft.com/en-us/azure/azure-functions/functions-run-local#v2)

- Optional: [Batch Explorer](https://azure.github.io/BatchExplorer/) for monitoring batch jobs


## Serverless Reverse-Time Migration

Follow these steps to reproduce the RTM example:

 1. Optional: (Re-)Build the docker images and upload them to the Azure Container registry or use pre-built public images (we will add public pre-built images to docker hub).
    
     - First, build the base image locally:

     ``` 
     cd ~/Azure2019/src/AzureBatch/docker/base_image
     docker build -t devito_azure_base:v1.0 .
     ```

    - Next, we can build the TTI image locally as follows:

     ``` 
     cd ~/Azure2019/src/AzureBatch/docker/tti_image
     docker build -t devito_azure_tti:v1.0 .
     ```

    - Tag the image using your Azure repository name and push it to the container registry. E.g.:

     ```
     docker tag devito_azure_tti:v1.0 slimdockerwest.azurecr.io/devito_azure_tti:v1.0
     docker push slimdockerwest.azurecr.io/devito_azure_tti:v1.0
     ```

 2. Upload the model and acquisition geometry to Azure Blob Storage (we will add the model/geometry to an FTP server or a public blob so that it is available to everyone)

 3. Upload the application script to Blob Storage (`~/Azure2019/scripts/overthrust_3D_limited_offset.py`)

 4. Modify the Batch Shipyard config files (`~/Azure2019/src/AzureBatch/shipyard/config_intel`)

     - Add and fill out `credentials.yml`

     - Fill out missing entries in `config.yaml`, `jobs.yaml`, `pool.yaml` with corresponding pool size, VM types, etc.


5. Run the example using Batch Shipyard:

Move to corresponding directory: `cd ~/Azure2019/src/AzureBatch/shipyard/config_intel` or `cd ~/Azure2019/src/AzureBatch/shipyard/config_gcc`.

```
# Start pool
./shipyard pool add -v
￼
# Start job
./shipyard jobs add --tail stdout.txt -v
```

6. Optional: Monitor batch job using Batch Explorer


7. After job has completed, clean up:

```
# Delete job
./shipyard jobs del -v

# Shut down pool
./shipyard pool del -v
```

## Event-driven workflow and image summation

**More documentation to be added soon.**

 - Azure Function for event-driven image summation:  `~/Azure2019/src/AzureFunctions/QueueTrigger`

 - Iterator for iterative workflows (e.g. least squares RTM): `~/Azure2019/src/AzureFunctions/Iterator`
 
 - Invokation of Azure Batch job through Azure Functions: `~/Azure2019/src/AzureFunctions/ComputeGradient` (unfinished)

