# Cube Studio

English | [简体中文](README_CN.md)

### Infra


![image](https://user-images.githubusercontent.com/20157705/167534673-322f4784-e240-451e-875e-ada57f121418.png)

cube-studio is a one-stop cloud-native machine learning platform open sourced by Tencent Music, Currently mainly includes the following functions
 - 1、data management：Feature Store: Online and offline features; Dataset management, structure data and media data, Data Label Platform
 - 2、develop: notrbook(vscode/jupyter); docker Image management; image build online
 - 3、train：Pipeline Drag and drop online; Open Template Market; Distributed computing/training tasks, example tf/pytorch/mxnet/spark/ray/horovod/kaldi/volcano; batch priority scheduling; Resource Monitoring Alarm Balancing; Cron Scheduling
 - 4、Hyperparameter Search：nni, katib, ray
 - 5、inference：model manager; serverless traffic control; tf/pytorch/onnx/tensorrt model deploy, tfserving/torchserver/onnxruntime/triton inference; VGPU; Load Balancing、High availability、Elastic scaling
 - 6、infra：Multi-user;Multi-project; Multi-cluster; Edge Cluster Mode; blockchain sharing;

# Doc

https://github.com/tencentmusic/cube-studio/wiki

# WeChat group

 learning、deploy、experience、contributions join group, wechart id luanpeng1234 remark<open source>, [construction guide](https://github.com/tencentmusic/cube-studio/wiki/%E5%85%B1%E5%BB%BA%E6%8C%87%E5%8D%97)

<img border="0" width="20%" src="https://luanpeng.oss-cn-qingdao.aliyuncs.com/github/wechat.jpg" />
 
# Job Template

tips：
- 1、Easy to develop and more suitable for your own scenarios

| template  | type | describe |
| :----- | :---- | :---- |
| linux | base | Custom stand-alone operating environment, free to implement all custom stand-alone functions | 
| datax | import export | Import and export of heterogeneous data sources | 
| media-download | data processing | 	Distributed download of media files  | 
| video-audio | data processing | 	Distributed extraction of audio from video  | 
| video-img | data processing | Distributed extraction of pictures from video | 
| sparkjob | data processing | spark serverless |
| ray | data processing | python ray multi-machine distributed framework |
| volcano | data processing | volcano multi-machine distributed framework  | 
| xgb | machine learning | xgb model training and inference |
| ray-sklearn | machine learning | sklearn based on ray framework supports multi-machine distributed parallel computing  |
| pytorchjob-train | model train | 	Multi-machine distributed training of pytorch  | 
| horovod-train | model train | 	Multi-machine distributed training of horovod  | 
| tfjob | model train |  Multi-machine distributed training of tensorflow | 
| tfjob-train | model train | distributed training of tensorflow: plain and  runner  | 
| tfjob-runner | model train | distributed training of tensorflow: runner method  | 
| tfjob-plain | model train | distributed training of tensorflow: plain method | 
| kaldi-train | model train | Multi-machine distributed training of kaldi  | 
| tf-model-evaluation | model evaluate | distributed model evaluation of tensorflow2.3  | 
| tf-offline-predict | model inference | distributed offline model inference of tensorflow2.3  | 
| model-offline-predict | model inference |  distributed offline model inference of framework  | 
| deploy-service | model deploy | deploy inference service  | 

 
# Deploy

[wiki](https://github.com/tencentmusic/cube-studio/wiki/%E5%B9%B3%E5%8F%B0%E5%8D%95%E6%9C%BA%E9%83%A8%E7%BD%B2)

![cube](https://user-images.githubusercontent.com/20157705/174762561-29b18237-7d45-417e-b7c0-14f5ef96a0e6.gif)


# Contributor

algorithm：
@hujunaifuture <img width="5%" src="https://avatars.githubusercontent.com/u/19547589?v=4" />
@jaffe-fly <img width="5%" src="https://avatars.githubusercontent.com/u/49515380?s=96&v=4" />
@JLWLL  <img width="5%" src="https://avatars.githubusercontent.com/u/86763551?s=96&v=4" />
@ma-chengcheng<img width="5%" src="https://avatars.githubusercontent.com/u/15444349?s=96&v=4" />
@chendile <img width="5%" src="https://avatars.githubusercontent.com/u/42484658?s=96&v=4" />

platform：
@xiaoyangmai <img width="5%" src="https://avatars.githubusercontent.com/u/10969390?s=96&v=4" />
@VincentWei2021 <img width="5%" src="https://avatars.githubusercontent.com/u/77832074?v=4" />
@SeibertronSS <img width="5%" src="https://avatars.githubusercontent.com/u/69496864?v=4" />
@cyxnzb <img width="5%" src="https://avatars.githubusercontent.com/u/51886383?s=88&v=4" /> 
@gilearn <img width="5%" src="https://avatars.githubusercontent.com/u/107160156?s=88&v=4" />
@wulingling0108 <img width="5%" src="https://avatars.githubusercontent.com/u/45533757?v=4" />
<br>
<br>

# Company

![image](https://user-images.githubusercontent.com/20157705/176909239-f24cbf8d-8fb5-4326-abed-6fbc3f5a2d1f.png)
