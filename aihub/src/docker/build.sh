
hubhost=ccr.ccs.tencentyun.com/cube-studio/aihub

# 构建基础镜像，cpu/gpu版本
docker build -t $hubhost:base --build-arg FROM_IMAGES=ubuntu:20.04  -f Dockerfile . && docker push $hubhost:base &
docker build -t $hubhost:base-cuda11.4 --build-arg FROM_IMAGES=nvidia/cuda:11.4.0-runtime-ubuntu20.04  -f Dockerfile . && docker push $hubhost:base-cuda11.4 &

# 构建python镜像，cpu版本
array=( 3.9 3.8 3.6 )

for element in ${array[*]}
do
  echo $element
  condaenv=py${element/./}
  echo $condaenv
  docker build -t $hubhost:base-python$element --build-arg PYTHONVERSION=$element --build-arg CONDAENV=$condaenv -f Dockerfile-python . && docker push $hubhost:base-python$element &
done

# 构建python镜像，gpu版本
array=( 3.9 3.8 3.6 )

for element in ${array[*]}
do
  echo $element
  condaenv=py${element/./}
  echo $condaenv
  docker build -t $hubhost:base-cuda11.4-python$element --build-arg PYTHONVERSION=$element --build-arg CONDAENV=$condaenv -f Dockerfile-cuda11.4-python . && docker push $hubhost:base-cuda11.4-python$element &
done

wait

# 构建相应的notebook镜像

array=( base base-cuda11.4 base-python3.6 base-python3.8 base-python3.9 base-cuda11.4-python3.6 base-cuda11.4-python3.8 base-cuda11.4-python3.9 )

for element in ${array[*]}
do
  echo $element
  docker build -t $hubhost:$element-notebook --build-arg FROM_IMAGES=$hubhost:$element -f Dockerfile-notebook . && docker push $hubhost:$element-notebook &
done

wait

#docker build -t $hubhost:base-python3.8 --build-arg FROM_IMAGES=$hubhost:base-python3.8 -f Dockerfile-notebook .
#docker push $hubhost:base-python3.8
