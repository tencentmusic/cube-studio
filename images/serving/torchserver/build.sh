#set -ex
hubhost=ccr.ccs.tencentyun.com/cube-studio

#arr=("torchserve:0.4.2-cpu" "torchserve:0.4.2-gpu" "torchserve:0.5.3-cpu" "torchserve:0.5.3-gpu" "torchserve:0.6.0-cpu" "torchserve:0.6.0-gpu")
arr=("torchserve:0.7.1-cpu" "torchserve:0.7.1-gpu" "torchserve:0.8.2-cpu" "torchserve:0.8.2-gpu" "torchserve:0.9.0-cpu" "torchserve:0.9.0-gpu")

for value in ${arr[@]}
do
    echo $value
    docker build --network=host -t $hubhost/$value --build-arg FROM_IMAGES=pytorch/$value .
    docker push $hubhost/$value
done




