#!/bin/bash

if [ ! -d stablehlo ]
then
    echo "The stablehlo directory is missing"
    echo "To prepare tests, install stablehlo: https://github.com/openxla/stablehlo"
    exit
fi

if [ ! -f stablehlo/build/bin/stablehlo-opt ]
then
    echo "Missing: stablehlo/build/bin/stablehlo-opt"
    exit
fi

shopt="stablehlo/build/bin/stablehlo-opt -mlir-print-op-generic -split-input-file"

interpret_test=stablehlo/stablehlo/tests/interpret

if [ ! -d $interpret_test ]
then
    echo "Missing tests: $interpret_test"
    exit
fi

for test in `ls $interpret_test/*.mlir`
do
    name=testdata_`basename $test`
    $shopt $test > Tests/$name
done

test_data=stablehlo/stablehlo/testdata

if [ ! -d $test_data ]
then
    echo "Missing tests: $test_data"
    exit
fi

for test in `ls $test_data/*.mlir`
do
    name=testdata_`basename $test`
    $shopt $test > Tests/$name
done
