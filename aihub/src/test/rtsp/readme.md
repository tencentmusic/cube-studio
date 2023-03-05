
启动服务端

docker run --rm --name rtsp -it -v $PWD/rtsp-simple-server.yml:/rtsp-simple-server.yml -v $PWD:/app -p 8554:8554 aler9/rtsp-simple-server


将mp4转到服务端

ffmpeg -re -stream_loop -1 -i video_heng.mp4 -f rtsp -rtsp_transport tcp rtsp://host.docker.internal:8554/live.stream

打开浏览器查看模型视频流推理效果

http://127.0.0.1:8080/yolov5/rtspcapture?rtsp_url=rtsp://host.docker.internal:8554/live.stream