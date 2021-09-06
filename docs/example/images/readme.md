# 在线构建镜像

![](../pic/tapd_20424693_1630748567_87.png)

# 修改默认python版本

	rm /usr/bin/python
	ln -s /usr/bin/python3.6 /usr/bin/python
	rm /usr/bin/pip
	ln -s /usr/bin/pip3 /usr/bin/pip
	pip install pip --upgrade
	
# 容器内使用git clone

	git clone [http://userName:password@](http://userName:password@/)链接
	示例
	git clone http://pengluan:xxxx@git.code.oa.com/tme-data-infra/dev.git


# ubuntu 容器基础工具的封装

	RUN apt update

	# 安装运维工具
	RUN apt install -y --force-yes --no-install-recommends vim apt-transport-https gnupg2 ca-certificates-java rsync jq  wget git dnsutils iputils-ping net-tools curl mysql-client locales zip

	# 安装python
	RUN apt install -y python3.6-dev python3-pip libsasl2-dev libpq-dev \
		&& ln -s /usr/bin/python3 /usr/bin/python \
		&& ln -s /usr/bin/pip3 /usr/bin/pip


	# 安装中文
	RUN apt install -y --force-yes --no-install-recommends locales ttf-wqy-microhei ttf-wqy-zenhei xfonts-wqy && locale-gen zh_CN && locale-gen zh_CN.utf8
	ENV LANG zh_CN.UTF-8
	ENV LC_ALL zh_CN.UTF-8
	ENV LANGUAGE zh_CN.UTF-8

	# 便捷操作
	RUN echo "alias ll='ls -alF'" >> /root/.bashrc && \
		echo "alias la='ls -A'" >> /root/.bashrc && \
		echo "alias vi='vim'" >> /root/.bashrc && \
		/bin/bash -c "source /root/.bashrc"

	# 安装其他工具
	### 安装kubectl
	RUN curl -LO https://dl.k8s.io/release/v1.16.0/bin/linux/amd64/kubectl && chmod +x kubectl && mv kubectl /usr/local/bin/
	### 安装mysql客户端
	RUN apt install -y mysql-client-5.7
	### 安装java
	RUN apt install -y openjdk-8-jdk
	### 安装最新版的nodejs
	RUN curl -sL https://deb.nodesource.com/setup_13.x | bash -
	RUN apt-get install -y nodejs && npm config set unicode false



# 常用基础镜像

### ubuntu
	cuda10.1-cudnn7
		- ai.tencentmusic.com/tme-public/ubuntu-gpu:cuda10.1-cudnn7
		
	python3.6
		- ai.tencentmusic.com/tme-public/ubuntu-gpu:cuda10.1-cudnn7-python3.6
		
	python3.7
		- ai.tencentmusic.com/tme-public/ubuntu-gpu:cuda10.1-cudnn7-python3.7
		
	python3.8
		- ai.tencentmusic.com/tme-public/ubuntu-gpu:cuda10.1-cudnn7-python3.8
		
	cuda10.1-cuda10.0-cuda9.0-cudnn7.6
		- ai.tencentmusic.com/tme-public/gpu:ubuntu18.04-python3.6-cuda10.1-cuda10.0-cuda9.0-cudnn7.6-base

### tlinux
	- mirrors.tencent.com/star_library/tlinux-64bit-v2.2.20170418:latest

### python2.7
##### cuda
	- mirrors.tencent.com/star_library/g-tlinux2.2-python2.7-cuda9.0-cudnn7.6:latest
	- mirrors.tencent.com/star_library/g-tlinux2.2-python2.7-cuda10.0-cudnn7.4:latest
	- mirrors.tencent.com/star_library/g-tlinux2.2-python2.7-cuda10.0-cudnn7.6:latest
	- mirrors.tencent.com/star_library/g-tlinux2.2-python2.7-cuda10.1-cudnn7.6:latest
##### ttensorflow
	- mirrors.tencent.com/star_library/g-tlinux2.2-python2.7-cuda10.1-cudnn7.6-nccl2.5.6-ttf1.15.2.1:latest
##### tensorflow
	- mirrors.tencent.com/star_library/g-tlinux2.2-python2.7-cuda9.0-cudnn7.6-tf1.12:latest
	- mirrors.tencent.com/star_library/g-tlinux2.2-python2.7-cuda10.0-cudnn7.4-tf1.13:latest
	- mirrors.tencent.com/star_library/g-tlinux2.2-python2.7-cuda10.0-cudnn7.4-tf1.14:latest
	- mirrors.tencent.com/star_library/g-tlinux2.2-python2.7-cuda10.0-cudnn7.6-tf1.15:latest
	- mirrors.tencent.com/star_library/g-tlinux2.2-python2.7-cuda10.0-cudnn7.6-tf2.0:latest
	- mirrors.tencent.com/star_library/g-tlinux2.2-python2.7-cuda10.1-cudnn7.6-tf2.1:latest
##### pytorch1.4
	- mirrors.tencent.com/star_library/g-tlinux2.2-python2.7-cuda10.1-cudnn7.6-pytorch1.4-torchvision0.5:latest
##### horovod
	- mirrors.tencent.com/star_library/g-tlinux2.2-python2.7-cuda9.0-cudnn7.6-tf1.12-openmpi4.0.3-nccl2.5.6-ofed4.6-horovod:latest
	- mirrors.tencent.com/star_library/g-tlinux2.2-python2.7-cuda10.0-cudnn7.4-tf1.13-openmpi4.0.3-nccl2.5.6-ofed4.6-horovod:latest
	- mirrors.tencent.com/star_library/g-tlinux2.2-python2.7-cuda10.0-cudnn7.4-tf1.14-openmpi4.0.3-nccl2.5.6-ofed4.6-horovod:latest
	- mirrors.tencent.com/star_library/g-tlinux2.2-python2.7-cuda10.0-cudnn7.6-tf2.0-openmpi4.0.3-nccl2.5.6-ofed4.6-horovod:latest
	- mirrors.tencent.com/star_library/g-tlinux2.2-python2.7-cuda10.1-cudnn7.6-tf2.1-openmpi4.0.3-nccl2.5.6-ofed4.6-horovod:latest
	- mirrors.tencent.com/star_library/g-tlinux2.2-python2.7-cuda10.1-cudnn7.6-pytorch1.4-torchvision0.5-openmpi4.0.3-nccl2.5.6-ofed4.6-horovod:latest

### python3.6
##### cuda
	- mirrors.tencent.com/star_library/g-tlinux2.2-python3.6-cuda9.0-cudnn7.6:latest
	- mirrors.tencent.com/star_library/g-tlinux2.2-python3.6-cuda10.0-cudnn7.4:latest
	- mirrors.tencent.com/star_library/g-tlinux2.2-python3.6-cuda10.0-cudnn7.6:latest
	- mirrors.tencent.com/star_library/g-tlinux2.2-python3.6-cuda10.1-cudnn7.6:latest
##### cuda10.1-cuda10.0-cuda9.0-cudnn7.6
	- csighub.tencentyun.com/tme-kubeflow/gpu:python3.6-cuda10.1-cuda10.0-cuda9.0-cudnn7.6-base
 
##### ttensorflow
	- mirrors.tencent.com/star_library/g-tlinux2.2-python3.6-cuda10.1-cudnn7.6-nccl2.5.6-ttf1.15.2.1:latest
#### tensorflow
	- mirrors.tencent.com/star_library/g-tlinux2.2-python3.6-cuda9.0-cudnn7.6-tf1.12:latest
	- mirrors.tencent.com/star_library/g-tlinux2.2-python3.6-cuda10.0-cudnn7.4-tf1.13:latest
	- mirrors.tencent.com/star_library/g-tlinux2.2-python3.6-cuda10.0-cudnn7.4-tf1.14:latest
	- mirrors.tencent.com/star_library/g-tlinux2.2-python3.6-cuda10.0-cudnn7.6-tf1.15:latest
	- mirrors.tencent.com/star_library/g-tlinux2.2-python3.6-cuda10.0-cudnn7.6-tf2.0:latest
	- mirrors.tencent.com/star_library/g-tlinux2.2-python3.6-cuda10.1-cudnn7.6-tf2.1:latest
##### pytorch
	- mirrors.tencent.com/star_library/g-tlinux2.2-python3.6-cuda10.1-cudnn7.6-pytorch1.4-torchvision0.5:latest
	- mirrors.tencent.com/star_library/g-tlinux2.2-python3.6-cuda10.1-cudnn7.6-pytorch1.5-torchvision0.6:latest
	- mirrors.tencent.com/star_library/g-tlinux2.2-python3.6-cuda10.1-cudnn7.6-pytorch1.6-torchvision0.7:latest
	- mirrors.tencent.com/star_library/g-tlinux2.2-python3.6-cuda10.2-cudnn8.0-pytorch1.6-torchvision0.7:latest
	- mirrors.tencent.com/star_library/g-tlinux2.2-python3.6-cuda10.2-cudnn8.0-pytorch1.5-torchvision0.6:latest
##### horovod
	- mirrors.tencent.com/star_library/g-tlinux2.2-python3.6-cuda9.0-cudnn7.6-tf1.12-openmpi4.0.3-nccl2.5.6-ofed4.6-horovod:latest
	- mirrors.tencent.com/star_library/g-tlinux2.2-python3.6-cuda10.0-cudnn7.4-tf1.13-openmpi4.0.3-nccl2.5.6-ofed4.6-horovod:latest
	- mirrors.tencent.com/star_library/g-tlinux2.2-python3.6-cuda10.0-cudnn7.4-tf1.14-openmpi4.0.3-nccl2.5.6-ofed4.6-horovod:latest
	- mirrors.tencent.com/star_library/g-tlinux2.2-python3.6-cuda10.0-cudnn7.6-tf2.0-openmpi4.0.3-nccl2.5.6-ofed4.6-horovod:latest
	- mirrors.tencent.com/star_library/g-tlinux2.2-python3.6-cuda10.1-cudnn7.6-tf2.1-openmpi4.0.3-nccl2.5.6-ofed4.6-horovod:latest
	- mirrors.tencent.com/star_library/g-tlinux2.2-python3.6-cuda10.1-cudnn7.6-pytorch1.4-torchvision0.5-openmpi4.0.3-nccl2.5.6-ofed4.6-horovod:latest

