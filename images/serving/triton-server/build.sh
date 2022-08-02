set -ex
hubhost=ccr.ccs.tencentyun.com/cube-studio

arr=("tritonserver:22.07-py3" "tritonserver:21.12-py3" "tritonserver:21.09-py3")

for value in ${arr[@]}
do
    echo $value
    docker build -t $hubhost/$value --build-arg FROM_IMAGES=nvcr.io/nvidia/$value .
    docker push $hubhost/$value
done



