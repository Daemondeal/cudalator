#!/bin/bash

if [ "$1" == "" ] ; then
    echo "USAGE: ./tools/run_demo folder"
    exit -1
fi

top_folder=$(realpath $1)

if [ ! -d "$top_folder" ] ; then
    echo "ERROR: Cannot find folder \"$top_folder\""
    exit -1
fi

set -e

demo_name=$(basename $top_folder)
sv_files="$top_folder/*.sv"
sim_main="$top_folder/main.cpp"
workdir="$PWD/private/$demo_name"

rm -rf $workdir
mkdir -p "$PWD/private"

# Compile the demo
cargo run -- $sv_files -o $workdir --cpu

# Sub in the demo simulation file
rm -f "$workdir/src/main.cpp"
cp -f $sim_main "$workdir/src/main.cpp"

# Run the demo
cd $workdir
make -j4
