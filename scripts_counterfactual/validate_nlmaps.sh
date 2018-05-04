#!/bin/sh

# path to nematus ( https://www.github.com/rsennrich/nematus )
nematus=/PATH/TO/NEMATUS

./validate_dev.sh $1 $2 &
./validate_test.sh $1 $2 &
wait
