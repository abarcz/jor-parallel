#!/bin/bash

for name in test_gpu_[0-9]*
do
	new_name=$(echo $name | sed -e 's/128/0128/g'| sed -e 's/256/0256/g'| sed -e 's/512/0512/g')
	mv $name $new_name
done
