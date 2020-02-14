FROM ubuntu:18.04
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

################################################# Core installs and ssh ###########################################
# set up base and ssh keys
COPY ssh_config /root/.ssh/config

RUN apt-get update \
		&& apt-get install -y software-properties-common \
		&& add-apt-repository -y ppa:jonathonf/gcc-9.2 \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends gcc-9 g++-9 gfortran \
        build-essential ca-certificates wget openssh-client openssh-server tzdata htop vim git-core net-tools\
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

################################################# Make gcc-9 the default gcch ###########################################
RUN rm /usr/bin/gcc && ln -s /usr/bin/gcc-9 /usr/bin/gcc
RUN rm /usr/bin/g++ && ln -s /usr/bin/g++-9 /usr/bin/g++

################################################# Install IB drivers #####################################################
# Librairies with devel versions
RUN apt-get install -y kmod lsof debhelper automake libnuma1 pkg-config ethtool autotools-dev \
                       libltdl-dev swig tk bison flex libnl-route-3-200 m4 dpatch libglib2.0-0 \
											 graphviz libnl-3-200 chrpath libmnl0 libelf1 tcl pciutils autoconf \
											 zlib1g-dev libnuma-dev

# MLNX_OFED installation
ENV OS_VER="ubuntu18.04"
ENV PLATFORM="x86_64"
ENV MOFED_VER="4.7-3.2.9.0"
RUN wget -q -O - http://content.mellanox.com/ofed/MLNX_OFED-${MOFED_VER}/MLNX_OFED_LINUX-${MOFED_VER}-${OS_VER}-${PLATFORM}.tgz | tar -xzf - \
    && mv ./MLNX_OFED_LINUX-${MOFED_VER}-${OS_VER}-${PLATFORM} /tmp/mlnx \
		&& ./tmp/mlnx/mlnxofedinstall --user-space-only --without-fw-update --all --force \
    && rm -rf /tmp/mlnx

#  HPC-X installation
ENV MOFED_VER="4.7-1.0.0.1"
ENV HPCX_VER="v2.5"
RUN wget -q -O - http://www.mellanox.com/downloads/hpc/hpc-x/${HPCX_VER}/hpcx-${HPCX_VER}.0-gcc-MLNX_OFED_LINUX-${MOFED_VER}-${OS_VER}-${PLATFORM}.tbz | tar -xjf - \
    && mv ./hpcx-${HPCX_VER}.0-gcc-MLNX_OFED_LINUX-${MOFED_VER}-${OS_VER}-${PLATFORM} /opt/hpcx
ENV HPCX_HOME=/opt/hpcx

################################################# IReinstall openmpi with HPC_X and MLNX #########################################
# Uncompress and recompile the HPC-x version of openmpi
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

################################################# Build MPI examples in case #####################################################
RUN source $HPCX_HOME/hpcx-mt-init.sh && hpcx_load && cd $HPCX_MPI_TESTS_DIR/examples && make

################################################# Env variable so that root allwoed to run in docker ##############################
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1


################################################# Clean apt-get ##############################
RUN rm -rf /var/lib/apt/lists/* \
	&& apt-get clean

############################################### Utility file for openmpi in batch shipyard #######################
COPY setup-ompi.sh /opt/setup-ompi.sh
