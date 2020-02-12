FROM mloubout/devito-azure:vIB

RUN apt update && apt-get install -y net-tools
# Install storage queue
RUN pip3 install azure-storage-blob h5py segyio boto3
RUN pip3 install azure-storage

# Add application
ADD ./tti /app/tti
COPY setup-ompi.sh /opt/setup-ompi.sh

# set up sshd on port 23
EXPOSE 23
CMD ["/usr/sbin/sshd", "-D", "-p", "23"]
