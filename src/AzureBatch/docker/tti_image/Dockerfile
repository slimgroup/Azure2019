FROM devito_azure_base:v1.0

# Install storage queue
RUN pip3 install azure-storage-blob==2.1.0 h5py segyio boto3

# Current sympy bug -> force version 1.5
RUN pip3 install sympy==1.5

# Add devito setup from JUDI, keep only Python part
RUN git clone --single-branch --branch examples https://github.com/slimgroup/JUDI.jl /app/JUDI.jl \
    && mv /app/JUDI.jl/src/pysource /app/. \
    && rm -rf /app/JUDI.jl

ADD ./AzureUtilities.py /app/pysource/.
ADD ./utils.py /app/pysource/.

# set up sshd on port 23
EXPOSE 23
CMD ["/usr/sbin/sshd", "-D", "-p", "23"]
