docker build -t ccr.ccs.tencentyun.com/cube-studio/aihub:app1 ./app1/ && docker push ccr.ccs.tencentyun.com/cube-studio/aihub:app1 &
docker build -t ccr.ccs.tencentyun.com/cube-studio/aihub:animegan ./animegan/ && docker push ccr.ccs.tencentyun.com/cube-studio/aihub:animegan &
docker build -t ccr.ccs.tencentyun.com/cube-studio/aihub:stable-diffusion ./stable-diffusion/ && docker push ccr.ccs.tencentyun.com/cube-studio/aihub:stable-diffusion &
docker build -t ccr.ccs.tencentyun.com/cube-studio/aihub:paddleocr ./paddleocr/ && docker push ccr.ccs.tencentyun.com/cube-studio/aihub:paddleocr &
docker build -t ccr.ccs.tencentyun.com/cube-studio/aihub:gfpgan ./gfpgan/ && docker push ccr.ccs.tencentyun.com/cube-studio/aihub:gfpgan &
docker build -t ccr.ccs.tencentyun.com/cube-studio/aihub:paddlespeech-asr ./paddlespeech-asr/ && docker push ccr.ccs.tencentyun.com/cube-studio/aihub:paddlespeech-asr &
docker build -t ccr.ccs.tencentyun.com/cube-studio/aihub:humanseg ./humanseg/ && docker push ccr.ccs.tencentyun.com/cube-studio/aihub:humanseg &
docker build -t ccr.ccs.tencentyun.com/cube-studio/aihub:paddlespeech-cls ./paddlespeech-cls/ && docker push ccr.ccs.tencentyun.com/cube-studio/aihub:paddlespeech-cls &
docker build -t ccr.ccs.tencentyun.com/cube-studio/aihub:ddddocr ./ddddocr/ && docker push ccr.ccs.tencentyun.com/cube-studio/aihub:ddddocr &
docker build -t ccr.ccs.tencentyun.com/cube-studio/aihub:paddlespeech-tts ./paddlespeech-tts/ && docker push ccr.ccs.tencentyun.com/cube-studio/aihub:paddlespeech-tts &
docker build -t ccr.ccs.tencentyun.com/cube-studio/aihub:yolov3 ./yolov3/ && docker push ccr.ccs.tencentyun.com/cube-studio/aihub:yolov3 &
docker build -t ccr.ccs.tencentyun.com/cube-studio/aihub:deoldify ./deoldify/ && docker push ccr.ccs.tencentyun.com/cube-studio/aihub:deoldify &
docker build -t ccr.ccs.tencentyun.com/cube-studio/aihub:panoptic ./panoptic/ && docker push ccr.ccs.tencentyun.com/cube-studio/aihub:panoptic &


wait