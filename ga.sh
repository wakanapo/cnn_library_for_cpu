#!/bin/bash

TIME=`date +%m%d%H%M`
mkdir $TIME
./bin/ga test --weights_output=$TIME/$1
