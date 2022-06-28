set -ex
hubhost=ccr.ccs.tencentyun.com/cube-studio

arr=("serving:1.11.0" "serving:1.11.0-gpu" "serving:1.12.0" "serving:1.12.0-gpu" "serving:1.13.0" "serving:1.13.0-gpu" "serving:1.14.0" "serving:1.14.0-gpu" "serving:2.0.0" "serving:2.0.0-gpu" "serving:2.1.4" "serving:2.1.4-gpu" "serving:2.2.3" "serving:2.2.3-gpu" "serving:2.3.4" "serving:2.3.4-gpu" "serving:2.4.3" "serving:2.4.3-gpu" "serving:2.5.2" "serving:2.5.2-gpu" "serving:2.6.0" "serving:2.6.0-gpu")

for value in ${arr[@]}
do
    echo $value
    docker build -t $hubhost/$value --build-arg FROM_IMAGES=tensorflow/$value .
    docker push $hubhost/$value
done



