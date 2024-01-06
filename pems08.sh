#!/bin/bash

function evalcmd () {

    echo $1

    eval $1

    sleep 0.5s

}


k=35
lweight=0.5
a_sg_nk=0.5
debug=0
dataset="pems08"
ada=1
unknown_ratio=0.5
split_type=(1 2 3 4)


for ((c=0; c<4; c++))
do
    wholecommand="python -u run_model.py --debug ${debug} --unknown_ratio ${unknown_ratio} --dataset ${dataset} --ada ${ada} --a_sg_nk ${a_sg_nk} --lweight ${lweight} --k ${k} --split_type ${split_type[$c]}"

    evalcmd "$wholecommand"
done
