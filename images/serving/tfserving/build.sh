set -ex
hubhost=ccr.ccs.tencentyun.com/cube-studio

#arr=("1.14.0" "1.14.0-gpu" "2.0.0" "2.0.0-gpu" "2.1.4" "2.1.4-gpu" "2.2.3" "2.2.3-gpu" "2.3.4" "2.3.4-gpu" "2.4.3" "2.4.3-gpu" "2.5.2" "2.5.2-gpu" "2.6.0" "2.6.0-gpu")

arr=("2.14.1-gpu" "2.14.1" "2.13.1-gpu" "2.13.1" "2.12.2-gpu" "2.12.2" "2.11.1-gpu" "2.11.1" "2.10.1-gpu" "2.10.1" "2.9.3-gpu" "2.9.3" "2.8.4-gpu" "2.8.4" "2.7.4-gpu" "2.7.4" "2.6.5-gpu" "2.6.5" "2.5.4-gpu" "2.5.4")

for value in ${arr[@]}
do
    echo $value
    docker build --network=host -t $hubhost/tfserving:$value --build-arg FROM_IMAGES=tensorflow/serving:$value .
    docker push $hubhost/tfserving:$value
done



