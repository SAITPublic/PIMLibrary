#!/usr/bin/env bash

SCRIPT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo SCRIPT_ROOT, $SCRIPT_ROOT
STAGE=rocm4.0
VERSION=tf2.4-latest-dev

pushd ${SCRIPT_ROOT}
docker build --build-arg http_proxy=$http_proxy \
             --build-arg https_proxy=$https_proxy \
             -t fim-${STAGE}:${VERSION} ${SCRIPT_ROOT} -f ${SCRIPT_ROOT}/$1
popd
