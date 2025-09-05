#!/bin/bash

# remote config
REMOTE_USER="jetson"
REMOTE_HOST="192.168.1.47"
REMOTE_DIR="~/cudalator_tests"


if [ "$1" == "" ] ; then
    echo "USAGE: $0 <folder> [--remote] [--cpu]"
    exit -1
fi
FOLDER=$1

REMOTE=false
CPU=false
if [[ " $@ " =~ " --remote " ]]; then
  REMOTE=true
fi
if [[ " $@ " =~ " --cpu " ]]; then
  CPU=true
fi

top_folder=$(realpath "$FOLDER")
if [ ! -d "$top_folder" ] ; then
    echo "ERROR: Cannot find folder \"$top_folder\""
    exit -1
fi

set -e

demo_name=$(basename "$top_folder")
sv_files="$top_folder/*.sv"
sim_main="$top_folder/main.cpp"
workdir="$PWD/private/$demo_name"
rm -rf "$workdir"
mkdir -p "$PWD/private"

CARGO_ARGS=""
if [ "$CPU" = true ]; then
  CARGO_ARGS="--cpu"
  echo "---> Generating CPU version ..."
else
  echo "---> Generating CUDA kernel ..."
fi
cargo run -- $sv_files -o "$workdir" $CARGO_ARGS

echo "---> Substituting the demo main.cpp ..."
rm -f "$workdir/src/main.cpp"
cp -f "$sim_main" "$workdir/src/main.cpp"


# Set CMake arguments for CUDA only if not a CPU build
CMAKE_ARGS=""
if [ "$CPU" = false ]; then
  CMAKE_ARGS="CMAKE_ARGS='-DENABLE_CUDA=ON'"
fi

if [ "$REMOTE" = true ]; then
  # Remote execution
  echo "---> Copying the project to ${REMOTE_HOST}"
  rsync -az --delete \
    --info=progress2,stats2,flist0,name0,del0 \
    "$workdir/" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/$(basename "$workdir")/"

  echo "---> Building and running on ${REMOTE_HOST}"
  ssh "${REMOTE_USER}@${REMOTE_HOST}" "export PATH=/usr/local/cuda/bin:\$PATH && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH && cd ${REMOTE_DIR}/$(basename "$workdir") && rm -rf build && make -j4 $CMAKE_ARGS"
else
  echo "---> Building and running locally ..."
  cd "$workdir"
  make -j4 $CMAKE_ARGS
fi
