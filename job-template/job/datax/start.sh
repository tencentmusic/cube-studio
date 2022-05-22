#!/bin/bash

while getopts f: flag
do
  case "${flag}" in
      f) file=${OPTARG};;
  esac
done

python datax.py $file

