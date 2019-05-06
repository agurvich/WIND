#!/bin/bash

## $1 (Katz96 or NR_test) $2 integer number of tilings

DATADIR=../data/${1}_${2}

cp ${DATADIR}/${1}_${2}_debug.txt ../cuda/debug/input${2}.h
echo "#"include '"'input${2}.h'"' > ../cuda/debug/debug.c
less ../cuda/debug/debug_suffix.c >> ../cuda/debug/debug.c
cd ../cuda/debug
make
