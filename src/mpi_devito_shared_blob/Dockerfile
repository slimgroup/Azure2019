FROM ubuntu:18.04

# set up base and ssh keys and install mpich
COPY ssh_config /root/.ssh/config 
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential ca-certificates wget openssh-client openssh-server \
        mpich libmpich-dev \
    && mkdir -p /var/run/sshd \
    && ssh-keygen -A \
    && sed -i 's/UsePAM yes/UsePAM no/g' /etc/ssh/sshd_config \
    && sed -i 's/#PermitRootLogin yes/PermitRootLogin yes/g' /etc/ssh/sshd_config \
    && sed -i 's/#RSAAuthentication yes/RSAAuthentication yes/g' /etc/ssh/sshd_config \
    && sed -i 's/#PubkeyAuthentication yes/PubkeyAuthentication yes/g' /etc/ssh/sshd_config \
    && ssh-keygen -f /root/.ssh/id_rsa -t rsa -N '' \
    && chmod 600 /root/.ssh/config \
    && chmod 700 /root/.ssh \
    && cp /root/.ssh/id_rsa.pub /root/.ssh/authorized_keys

# # Install gcc/g++
RUN apt-get -y install gcc-8 g++-8

# Install git for devito
RUN apt-get install -y git-core

# Python 3
RUN apt-get install -y python3 python3-pip

# Devito
RUN pip3 install --upgrade pip
RUN pip3 install --user git+https://github.com/opesci/devito.git@docker-save
RUN pip3 install --user segyio azure-storage-blob mpi4py

# set up sshd on port 23
EXPOSE 23
CMD ["/usr/sbin/sshd", "-D", "-p", "23"]