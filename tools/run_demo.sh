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


compute_version_string=""
type nvidia-smi
if [[ $? -eq 0 ]] ; then
  compute=$(nvidia-smi --query-gpu=compute_cap --format=noheader)

  echo "Detected compute version $compute"
  compute_version_string="-DCUDA_COMPUTE_VERSION=$($compute | tr -d '.')"
fi

set -e

demo_name=$(basename $top_folder)
sv_files="$top_folder/*.sv"
sim_main="$top_folder/main.cpp"
workdir="$PWD/private/$demo_name"

rm -rf $workdir
mkdir -p "$PWD/private"

if [[ -f "$top_folder/top.txt" ]] ; then
  top_entity="--top $(cat "$top_folder/top.txt")"
else
  top_entity=""
fi


# ---- CHANGED: generate CPU or GPU ----
if [ "$CPU" = true ]; then
  echo "---> Generating CPU version ..."
  cargo run -- $sv_files -o $workdir --cpu $top_entity
else
  echo "---> Generating GPU version ..."
  cargo run -- $sv_files -o $workdir $top_entity
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
  make -j4 CMAKE_ARGS="-DENABLE_CUDA=ON $cuda_define $compute_version_string"
fi

