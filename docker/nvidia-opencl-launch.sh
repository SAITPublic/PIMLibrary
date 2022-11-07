#!/usr/bin/env bash
image_name=$1
container_name=pim-${image_name//[:\/]/-}-$(id -u -n $USER)
WORKSPACE=$HOME/$2

docker ps -a | grep ${container_name} > /dev/null 2>&1
result=$?

if [ ! $result -eq 0 ]; then
    echo "No Container found, Create new containter ${container_name}"

    mkdir -p $WORKSPACE

    # Start docker images (only once)
      docker run --rm -it --gpus all --user root --network=host --ipc=host\
                 -e LOCAL_USER_ID=`id -u $USER` \
                 --security-opt seccomp:unconfined \
                 --cap-add=ALL --privileged \
                 -e DISPLAY=$DISPLAY \
                 -p 8080:8080 \
                 -v /dev:/dev \
                 -v /lib/modules:/lib/modules \
                 -v $HOME/.ssh:/home/user/.ssh \
                 -v /tmp/.X11-unix:/tmp/.X11-unix \
                 -v /root/.Xauthority:/root/.Xauthority:rw \
		 -v /lib/modules:/lib/modules \
		 -v $WORKSPACE:/home/user/pim-workspace \
		 --name=${container_name} \
                 $image_name /bin/bash
else
    docker start  ${container_name} && docker attach ${container_name}
fi

