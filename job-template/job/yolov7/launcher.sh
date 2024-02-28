#!/bin/bash
set -e

# Setup getopt.
long_opts="train:,val:,classes:,batch_size:,epoch:,weights:,save_model_path:"
getopt_cmd=$(getopt -o da: --long "$long_opts" \
            -n $(basename $0) -- "$@") || \
            { echo -e "\nERROR: Getopt failed. Extra args\n"; exit 1;}

eval set -- "$getopt_cmd"
while true; do
    case "$1" in
        -t|--train) echo "train is $2" && train=$2;;
        -v|--val) echo "val is $2" && val=$2;;
        -c|--classes) echo "classes is $2" && classes=$2;;
        -b|--batch_size) echo "batch_size is $2" && batch_size=$2;;
        -e|--epoch) echo "epoch is $2" && epoch=$2;;
        -w|--weights) echo "weights is $2" && weights=$2;;
        -s|--save_model_path) echo "save_model_path is $2" && save_model_path=$2;;
        --) shift; break;;
    esac
    shift
done


python3 save_config.py --train $train --val $val --classes $classes

python train.py --weights $weights --cfg /yolov7/yolov7.yaml --data /yolov7/data.yaml --batch-size $batch_size --epoch $epoch

rm -rf $save_model_path

cp /yolov7/runs/train/exp2/weights/best.pt $save_model_path
