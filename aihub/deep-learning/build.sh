docker build -t ccr.ccs.tencentyun.com/cube-studio/aihub:paddlespeech-tts ./paddlespeech-tts/ && docker push ccr.ccs.tencentyun.com/cube-studio/aihub:paddlespeech-tts &
docker build -t ccr.ccs.tencentyun.com/cube-studio/aihub:paddlespeech-asr ./paddlespeech-asr/ && docker push ccr.ccs.tencentyun.com/cube-studio/aihub:paddlespeech-asr &
docker build -t ccr.ccs.tencentyun.com/cube-studio/aihub:paddlespeech-cls ./paddlespeech-cls/ && docker push ccr.ccs.tencentyun.com/cube-studio/aihub:paddlespeech-cls &

wait