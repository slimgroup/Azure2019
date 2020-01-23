FROM devito_azure_base:v1.0

# Install storage queue
RUN pip3 install --user azure-storage-queue h5py

# Add application
ADD ./tti /app/tti

# set up sshd on port 23
EXPOSE 23
CMD ["/usr/sbin/sshd", "-D", "-p", "23"]
