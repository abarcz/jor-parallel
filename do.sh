#!/bin/bash

array=( 128 256 512 1024 2048 4096 8192 )
for SIZE in "${array[@]}"
do
	./main $SIZE 10 1 > "results-floats/test_gpu_$SIZE"
done
