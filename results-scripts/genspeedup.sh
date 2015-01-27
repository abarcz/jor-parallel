#!/bin/bash

for size in 128 256 512 1024 2048 4096 8192
do
	texline="$size"
	time=""
	for name in *_$size
	do
		res=`grep "Time" $name`
		tex=$(echo $res | sed -e 's/.*mean \([0-9.]*\).*std \([0-9.]*\).*/\1/g')
		if [ "$time" == "" ]; then
			time=$tex
			#echo "setting time to $time"
		else
			speedup=$(echo "scale=2; $time/$tex" | bc)
			texline="$texline & $speedup"
		fi
	done
	echo "$texline \\\\"
done
