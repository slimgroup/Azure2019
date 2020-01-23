FROM ubuntu:18.04

# set up base and ssh keys
COPY ssh_config /root/.ssh/config
RUN apt-get update
RUN apt-get install -y g++-8  make ca-certificates wget openssh-server openssh-client \
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

# Utils
RUN apt-get install -y htop vim git-core

# download and install OpenMPI
ENV MPICH_VERSION="3.3"
RUN wget -q -O - wget https://www.mpich.org/static/downloads/${MPICH_VERSION}/mpich-${MPICH_VERSION}.tar.gz | tar -xzf - \
    && cd mpich-${MPICH_VERSION} \
    && ./configure --enable-mpirun-prefix-by-default --disable-fortran CC=gcc-8 CXX=g++-8 \
    && make -j 8 && make install \
    && cd .. \
    && rm -rf mpich-${MPICH_VERSION}

# Devito
RUN apt-get install -y python3 python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade git+https://github.com/opesci/devito.git
RUN pip3 install segyio azure-storage-blob mpi4py h5py