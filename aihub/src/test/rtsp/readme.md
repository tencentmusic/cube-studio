
启动服务端

docker run --rm --name rtsp -it -v $PWD/rtsp-simple-server.yml:/rtsp-simple-server.yml -v $PWD:/app -p 8554:8554 aler9/rtsp-simple-server


将mp4转到服务端

ffmpeg -re -stream_loop -1 -i video_heng.mp4 -f rtsp -rtsp_transport tcp rtsp://host.docker.internal:8554/live.stream