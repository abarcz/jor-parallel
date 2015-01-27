#!/bin/bash


echo "["
for size in 128 256 512 1024 2048 4096 8192
do
	texline="$size,"
	for name in *_$size
	do
		res=`grep "Time" $name`
		tex=$(echo $res | sed -e 's/.*mean \([0-9.]*\).*std \([0-9.]*\).*/\1/g')
		texline="$texline $tex,"
	done
	echo "$(echo "$texline;" | sed -e 's/,;/;/g')"
done
echo "]"
