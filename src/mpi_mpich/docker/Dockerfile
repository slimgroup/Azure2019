FROM ubuntu:18.04

# set up base and ssh keys
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

RUN apt-get -y install gcc-8 g++-8
RUN apt install -y python3 python3-pip
RUN apt-get install -y vim git-core

# Install devito
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade --user git+https://github.com/opesci/devito.git segyio azure-storage-blob

# Devito requirements
ADD ./devito_isotropic/requirements.txt /app/requirements.txt
RUN pip3 install -U -r /app/requirements.txt
RUN pip3 install -U git+https://github.com/inducer/codepy mpi4py

# Compiler and default Devito environment variables
ENV DEVITO_ARCH="gcc"
ENV DEVITO_OPENMP="1"
ENV OMP_NUM_THREADS="2"
ENV PYTHONPATH "${PYTHONPATH}:/app/devito_isotropic"
ENV DEVITO_LOGGING="DEBUG"

# Add application
ADD ./devito_isotropic /app/devito_isotropic

# set up sshd on port 23
EXPOSE 23
CMD ["/usr/sbin/sshd", "-D", "-p", "23"]
