#!/bin/bash

USER_ID=${LOCAL_USER_ID:-9001}
USERENV="PATH=${PATH} LD_LIBRARY_PATH=${LD_LIBRARY_PATH} PYTHONPATH=${PYTHONPATH}"

echo "Starting with UID : $USER_ID"

if [ ! $USER_ID -eq 0 ]; then
        getent passwd $USER_ID > /dev/null 2>&1
        result=$?

        if [ ! $result -eq 0 ]; then
            echo "Create new uid"
            useradd --shell /bin/bash -u $USER_ID -o -M -c "" -d /home/user user-$USER_ID
            echo "user-$USER_ID ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
        fi
        usermod -a -G developer user-$USER_ID

        if [ ! -e /home/user ]; then
            mkdir -p /home/user
        fi

        chown user-$USER_ID:developer /home/user
        chmod 775 /home/user
	cp /tmp/bashrc /home/user/.bashrc

        export HOME=/home/user

       cd $HOME; sudo -E -u user-$USER_ID /usr/bin/env ${USERENV} "$@"
else
    exec "$@"
fi

