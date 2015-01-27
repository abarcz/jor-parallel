#!/bin/bash

for name in test_cpu_simd_[0-9]*
do
	new_name=$(echo $name | sed -e 's/simd/asimd/g')
	#echo $new_name
	mv $name $new_name
done
