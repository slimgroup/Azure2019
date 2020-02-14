FROM ubuntu:18.04
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

################################################# Core installs and ssh ###########################################
COPY ssh_config /root/.ssh/config

RUN apt-get update \
	  && apt-get install -y software-properties-common \
		&& add-apt-repository -y ppa:jonathonf/gcc-9.2 \
    && apt-get install -y --no-install-recommends \
        build-essential ca-certificates wget openssh-client openssh-server \
        mpich libmpich-dev gcc-9 g++-9 htop vim git-core\
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

# clean
RUN rm -rf /var/lib/apt/lists/* \
	&& apt-get clean
