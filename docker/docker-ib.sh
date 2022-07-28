#!/usr/bin/env bash
image_name=$1
container_name=ib-${image_name//[:\/]/-}-$(id -u -n $USER)
workspace=$HOME/$2
devices='--device=/dev/kfd --device=/dev/dri --device=/dev/infiniband/rdma_cm --device=/dev/infiniband/uverbs3 --device=/dev/infiniband/uverbs2 --device=/dev/infiniband/uverbs1 --device=/dev/infiniband/uverbs0 --device=/dev/infiniband/issm3 --device=/dev/infiniband/issm2 --device=/dev/infiniband/issm1 --device=/dev/infiniband/issm0 --device=/dev/infiniband/umad3 --device=/dev/infiniband/umad2 --device=/dev/infiniband/umad1 --device=/dev/infiniband/umad0'

docker ps -a | grep ${container_name} > /dev/null 2>&1
result=$?

if [ ! $result -eq 0 ]; then
    echo "No Container found, Create new containter ${container_name}"

    mkdir -p $WORKSPACE

    # Start docker images (only once)
    docker run -it --user root --network=host --uts=host --group-add video --ipc=host --shm-size 16G \
               --ulimit=stack=67108864 --ulimit=memlock=-1 \
               $devices \
               -e LOCAL_USER_ID=`id -u $USER` \
               --security-opt seccomp:unconfined \
               --cap-add=ALL --privileged \
               -e DISPLAY=$DISPLAY \
               -p 8080:8080 \
               --dns 10.41.128.98 \
               -v /dev:/dev \
               -v /lib/modules:/lib/modules \
               -v $HOME/.ssh:/home/user/.ssh \
               -v /tmp/.X11-unix:/tmp/.X11-unix \
               -v /root/.Xauthority:/root/.Xauthority:rw \
               -v /lib/modules:/lib/modules \
               -v $workspace:/home/user/pim-workspace \
               --name=${container_name} \
               $image_name /bin/bash
else
    docker start  ${container_name} && docker attach ${container_name}
fi

