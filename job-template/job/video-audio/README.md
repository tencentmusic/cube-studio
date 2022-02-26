当前代码当前三个job模板：分布式媒体文件下载、分布式视频抽帧、分布式视频收取音频

# media-download模板
需要input_file文件的内容符合固定的格式。  
url类型，每行格式：`$url $local_path`  

# video-img模板
分布式抽取视频帧，文本格式要为
`$local_video_path $des_img_dir $frame_rate`

# video-audio模板
分布式抽取音频，文本格式要为
`$local_video_path $des_audio_path`

