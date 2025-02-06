#!/bin/bash

if [ "$2" == "" ] ; then
    echo "USAGE: ./run_test.sh file top_entity"
    exit -1
fi

file=$(realpath $1)
top_entity=$2
name=$(basename $file .sv) 


if [ ! -f "$file" ] ; then
    echo "Cannot find file $file"
    exit -1
fi


outdir=$PWD/output
mkdir -p $outdir

set -e

rm -rf build
mkdir -p build
cd build

# QuestaSim
vlog $file
vsim -do "run -all" -do "quit" $top_entity -c > $outdir/questasim.log 

# Icarus Verilog
iverilog -g2012 $file -o test
vvp ./test > $outdir/icarus.log

# Verilator

cat > sim_main.cpp << EOM
#include <verilated.h>
#include "verilated_vcd_c.h"
#include "V$name.h"

int main(int argc, char **argv)
{
    // Construct context object, design object, and trace object
    VerilatedContext *m_contextp = new VerilatedContext; // Context
    V$name *m_duvp = new V$name;                 // Design
    // Trace configuration
    while (!m_contextp->gotFinish())
    {
        // Refresh circuit state
        m_duvp->eval();
        // Increase simulation time
        m_contextp->timeInc(1);
    }
    // Free memory
    delete m_duvp;
    return 0;
}
EOM

verilator -cc $file --build --exe --timing ./sim_main.cpp
"./obj_dir/V$name" > $outdir/verilator.log


# Printing
#
# ech
printf "\n### DONE ###"

printf "\nQuestaSim:\n"
grep -A 30000 "# run -all" $outdir/questasim.log

printf "\nVerilator:\n"
cat $outdir/verilator.log

printf "\nIcarus:\n"
cat $outdir/icarus.log
