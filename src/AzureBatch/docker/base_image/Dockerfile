FROM ubuntu:18.04
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

# set up base and ssh keys
COPY ssh_config /root/.ssh/config

RUN apt-get update \
		&& apt-get install -y software-properties-common \
		&&  add-apt-repository -y ppa:jonathonf/gcc-9.2 \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends gcc-9 g++-9 gfortran \
        build-essential ca-certificates wget openssh-client openssh-server tzdata \
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

RUN rm /usr/bin/gcc && ln -s /usr/bin/gcc-9 /usr/bin/gcc
RUN rm /usr/bin/g++ && ln -s /usr/bin/g++-9 /usr/bin/g++
# MPI requires

##### MPI install
RUN apt-get install -y kmod lsof debhelper automake libnuma1 pkg-config ethtool autotools-dev \
                       libltdl-dev swig tk bison flex libnl-route-3-200 m4 dpatch libglib2.0-0 \
											 graphviz libnl-3-200 chrpath libmnl0 libelf1 tcl pciutils autoconf

# mlnx (IB required)
ADD ./mlnx /tmp/mpi/mlnx
RUN /tmp/mpi/mlnx/mlnxofedinstall --user-space-only --without-fw-update --all --force \
    && rm -rf /tmp/mpi/mlnx

#  HPC-X (IB required)
ADD ./hpcx /opt/hpcx
ENV HPCX_HOME=/opt/hpcx

# download and install MPICH
RUN apt-get install zlib1g-dev libnuma-dev
RUN tar xfp ${HPCX_HOME}/sources/openmpi-gitclone.tar.gz
RUN cd openmpi-gitclone \
    && source $HPCX_HOME/hpcx-mt-init.sh && hpcx_load \
    && CC=gcc-9 CXX=g++-9 ./configure \
		               --with-ucx=$HPCX_HOME/ucx \
		               --with-hcoll=/$HPCX_HOME/hcoll \
		               --prefix=${HPCX_HOME}/ompi \
		               --with-platform=optimized \
		               --enable-mca-no-build=btl-uct \
		               --with-device=ch4:ucx 2>&1 | tee config-icc-output.log \
		    && make -j32 all 2>&1 | tee build_icc.log \
		    && make -j24 install 2>&1 | tee install_icc.log \
		    && cd .. \
		    && rm -rf openmpi-gitclone

RUN source $HPCX_HOME/hpcx-mt-init.sh && hpcx_load && cd $HPCX_MPI_TESTS_DIR/examples && make

ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
# Utils
RUN apt-get install -y htop vim git-core

# Devito
RUN apt-get install -y python3 python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install matplotlib
RUN pip3 install --no-cache-dir --upgrade --force-reinstall git+https://github.com/devitocodes/devito.git
RUN source $HPCX_HOME/hpcx-mt-init.sh && hpcx_load && pip3 install mpi4py

# clean
RUN rm -rf /var/lib/apt/lists/* \
	&& apt-get clean
