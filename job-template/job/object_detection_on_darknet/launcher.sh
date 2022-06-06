#!/bin/bash
set -e

# Setup getopt.
long_opts="train_cfg:,data_cfg:,weights:"
getopt_cmd=$(getopt -o da: --long "$long_opts" \
            -n $(basename $0) -- "$@") || \
            { echo -e "\nERROR: Getopt failed. Extra args\n"; exit 1;}

eval set -- "$getopt_cmd"
while true; do
    case "$1" in
        -t|--train_cfg) echo "train_cfg is $2" && train_cfg=$2;;
        -d|--data_cfg) echo "data_cfg is $2" && data_cfg=$2;;
        -w|--weights) echo "weights is $2" && weights=$2;;
        --) shift; break;;
    esac
    shift
done


python3 setup_args.py --train_cfg "$train_cfg" --data_cfg "$data_cfg" --weights "$weights"

/app/darknet/darknet detector train /app/darknet/cfg/data.cfg /app/darknet/cfg/train.cfg $weights 2>&1
