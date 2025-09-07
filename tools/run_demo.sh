#!/bin/bash

if [ "$2" == "" ] ; then
    echo "USAGE: ./tools/run_demo <cpu/gpu> folder "
    exit -1
fi

top_folder=$(realpath $2)

if [ ! -d "$top_folder" ] ; then
    echo "ERROR: Cannot find folder \"$top_folder\""
    exit -1
fi

if [[ "$1" == "gpu" ]] ; then
  CPU=false
elif [[ "$1" == "cpu" ]] ; then
  CPU=true
else
  echo "USAGE: ./tools/run_demo <cpu/gpu> folder"
  exit -1
fi

set -e

demo_name=$(basename $top_folder)
sv_files="$top_folder/*.sv"
sim_main="$top_folder/main.cpp"
workdir="$PWD/private/$demo_name"

rm -rf $workdir
mkdir -p "$PWD/private"

# ---- CHANGED: generate CPU or GPU ----
if [ "$CPU" = true ]; then
  echo "---> Generating CPU version ..."
  cargo run -- $sv_files -o $workdir --cpu
else
  echo "---> Generating GPU version ..."
  cargo run -- $sv_files -o $workdir
fi

# ---- CHANGED: drop in the right main (cpp for CPU, prefer cu for GPU) ----
rm -f "$workdir/src/main.cpp" "$workdir/src/main.cu"
if [ "$CPU" = true ]; then
  cp -f $sim_main "$workdir/src/main.cpp"
else
  sim_main_cu="${sim_main%.cpp}.cu"
  if [ -f "$sim_main_cu" ]; then
    cp -f $sim_main_cu "$workdir/src/main.cu"
  else
    # fallback if you only have main.cpp
    cp -f $sim_main "$workdir/src/main.cpp"
  fi
fi

# ---- CHANGED: build (enable CUDA when not CPU) ----
cd $workdir
if [ "$CPU" = true ]; then
  make -j4
else
  make -j4 CMAKE_ARGS=-DENABLE_CUDA=ON
fi

