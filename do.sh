#!/bin/bash

for i in 1 2 3 4 5 6 7 8 9 10
do
	./main | grep "seconds" | sed "s/seconds: \([0-9.]*\)/\1/"
done
