#!/bin/bash

## $1 (Katz96 or NR_test) $2 integer number of tilings


NAME=${1}_neqntile_${2}_nsystemtile_${3}_fixed_${4}
DATADIR=../data/${NAME}

cp ${DATADIR}/${NAME}_debug.txt ../cuda/debug/input${1}_${2}_${3}_${4}.h
cp ${DATADIR}/${NAME}_debug.txt ../cuda/c_baseline/debug/input${1}_${2}_${3}_${4}.h

## make the gpu debug file
cd ../cuda/debug
echo "#"include '"'input${1}_${2}_${3}_${4}.h'"' > debug_auto.c
less debug_suffix.c >> debug_auto.c
make DEBUGFILE="debug_auto.c"


## make the cpu debug file
cd ../c_baseline/debug
echo "#"include '"'input${1}_${2}_${3}_${4}.h'"' > debug_auto.c
less debug_suffix.c >> debug_auto.c
make DEBUGFILE="debug_auto.c"
