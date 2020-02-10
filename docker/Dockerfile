FROM ubuntu:16.04
MAINTAINER Sudhakar M K <sudhakar.mk@samsung.com>

# Build Requirements
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends
RUN apt-get update
RUN apt-get --yes --allow-unauthenticated install sudo
RUN apt-get --yes --allow-unauthenticated install wget
RUN apt-get --yes --allow-unauthenticated install kmod

# Add rocm apt repository
RUN wget -qO - http://repo.radeon.com/rocm/apt/debian/rocm.gpg.key | sudo apt-key add -
RUN echo 'deb [arch=amd64] http://repo.radeon.com/rocm/apt/debian/ xenial main' | sudo tee /etc/apt/sources.list.d/rocm.list

RUN apt-get update
# Instal hipcc 
RUN apt-get install --yes --allow-unauthenticated hip-hcc

# Developement Requirments
RUN apt-get --yes --allow-unauthenticated install build-essential
RUN apt-get --yes --allow-unauthenticated install git
RUN apt-get --yes --allow-unauthenticated install vim
RUN apt-get --yes --allow-unauthenticated install byobu
RUN apt-get --yes --allow-unauthenticated install bash-completion
RUN apt-get --yes --allow-unauthenticated install clang-format-3.9
RUN apt-get --yes --allow-unauthenticated install cmake

# FIM Library requirements
RUN apt-get --yes --allow-unauthenticated install libgoogle-glog-dev

ENV QWORKSPACE=/home/user/fim-workspace
ENV ROCM_PATH=/opt/rocm
ENV HIP_PATH=${ROCM_PATH}/hip
ENV HCC_HOME=${ROCM_PATH}/hcc
ENV HIP_PLATFORM=hcc
ENV PATH=${ROCM_PATH}/bin:${ROCM_PATH}/bin:${ROCM_PATH}/hcc/bin:${ROCM_PATH}/hip/bin:${ROCM_PATH}/opencl/bin:$PATH
ENV LD_LIBRARY_PATH=${ROCM_PATH}/libhsakmt/lib:${ROCM_PATH}/hsa/lib:${ROCM_PATH}/hcc/lib:${ROCM_PATH}/hcc/compiler/lib:${ROCM_PATH}/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

RUN groupadd -g 9000 developer

COPY ./entrypoint.sh /usr/local/bin/entrypoint.sh
COPY ./bashrc /tmp/bashrc

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

CMD ["bash", "-l"]
