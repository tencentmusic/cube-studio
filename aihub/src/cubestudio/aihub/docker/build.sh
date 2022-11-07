docker build -t ccr.ccs.tencentyun.com/cube-studio/aihub:base-python3.9  -f Dockerfile-python3.9 . && docker push ccr.ccs.tencentyun.com/cube-studio/aihub:base-python3.9 &
docker build -t ccr.ccs.tencentyun.com/cube-studio/aihub:base-python3.8  -f Dockerfile-python3.8 . && docker push ccr.ccs.tencentyun.com/cube-studio/aihub:base-python3.8 &
docker build -t ccr.ccs.tencentyun.com/cube-studio/aihub:base-python3.6  -f Dockerfile-python3.6 . && docker push ccr.ccs.tencentyun.com/cube-studio/aihub:base-python3.6 &
docker build -t ccr.ccs.tencentyun.com/cube-studio/aihub:base-cuda11.4-python3.9  -f Dockerfile-cuda11.4-python3.9 . && docker push ccr.ccs.tencentyun.com/cube-studio/aihub:base-cuda11.4-python3.9 &

docker build -t ccr.ccs.tencentyun.com/cube-studio/aihub:base-cuda11.4  -f Dockerfile-cuda11.4 . && docker push ccr.ccs.tencentyun.com/cube-studio/aihub:base-cuda11.4 &
docker build -t ccr.ccs.tencentyun.com/cube-studio/aihub:base  -f Dockerfile . && docker push ccr.ccs.tencentyun.com/cube-studio/aihub:base &

wait