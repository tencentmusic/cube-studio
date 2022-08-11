#!/bin/bash
set -e

# Setup getopt.
long_opts="command:"
getopt_cmd=$(getopt -o da: --long "$long_opts" \
            -n $(basename $0) -- "$@") || \
            { echo -e "\nERROR: Getopt failed. Extra args\n"; exit 1;}

eval set -- "$getopt_cmd"
while true; do
    case "$1" in
        -c|--command) echo "command is $2" && command=$2;;
        --) shift; break;;
    esac
    shift
done

$command
