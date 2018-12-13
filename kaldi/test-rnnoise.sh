#!/bin/bash

wavDir=$1
wavDenoiseDir=$2

mkdir -p $wavDenoiseDir

for file in `ls $wavDir`
do
    ext=${file##*.}
    if [ $ext == "wav" ]
    then
        `./build/denoise/denoise-rnn/denoise-rnn $wavDir/$file $wavDenoiseDir/${file}_denose`
    fi
done


