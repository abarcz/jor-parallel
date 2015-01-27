#!/bin/bash

for size in 128 256 512 1024 2048 4096 8192
do
	texline="$size"
	for name in *_$size
	do
		res=`grep "RMSE: mean" $name`
		tex=$(echo $res | sed -e 's/.*mean \([0-9.]\{7\}\).*std \([0-9.]\{7\}\).*/ \1 (\2)/g')
		texline="$texline & $tex"
	done
	echo "$texline \\\\"
done
