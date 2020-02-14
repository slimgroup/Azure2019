FROM mloubout/devito-azure:vIB

# Devito
RUN apt update && apt-get install -y python3 python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install matplotlib
RUN pip3 install --no-cache-dir --upgrade --force-reinstall git+https://github.com/devitocodes/devito.git
RUN source $HPCX_HOME/hpcx-mt-init.sh && hpcx_load && pip3 install mpi4py
# RUN pip3 install mpi4py # Use this one for a base withotu HPCX installed

# Install storage queue
RUN pip3 install azure-storage-blob h5py segyio boto3
RUN pip3 install azure-storage

# Add application
ADD ./tti /app/tti


################################################# Clean apt-get ##############################
RUN rm -rf /var/lib/apt/lists/* \
	&& apt-get clean

# set up sshd on port 23
EXPOSE 23
CMD ["/usr/sbin/sshd", "-D", "-p", "23"]
