
# 卸载之前的

ubuntu 卸载 gpu包
```bash
sudo apt-get --purge remove -y *nvidia*  
sudo apt autoremove  
sudo apt-get --purge remove -y "*cublas*" "cuda*"  
sudo apt-get --purge remove -y "*nvidia*"  
```

centos 卸载gpu包

```bash
sudo yum remove nvidia-*  
rpm -qa|grep -i nvid|sort
yum  remove kmod-nvidia-*
sudo yum remove nvidia-*  

# sudo yum remove cublas-* cuda-*  
```

reboot    才能重新安装

# 安装新的

```bash
安装依赖
sudo apt-get install -y freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libgl1-mesa-glx libglu1-mesa libglu1-mesa-dev

# 或者安装520.61.05版本
wget https://us.download.nvidia.com/tesla/520.61.05/NVIDIA-Linux-x86_64-520.61.05.run
bash ./NVIDIA-Linux-x86_64-520.61.05.run

wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
bash ./cuda_11.8.0_520.61.05_linux.run

# 安装550的驱动
wget https://us.download.nvidia.com/tesla/550.54.14/NVIDIA-Linux-x86_64-550.54.14.run
bash ./NVIDIA-Linux-x86_64-550.54.14.run

wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run
bash .cuda_12.4.0_550.54.14_linux.run

```
安装后再重启

也可以使用命令推荐安装ubuntu-drivers devices，
```bash
sudo apt install -y nvidia-driver-535-server
重启机器
```

# fabricmanager 必须和驱动版本一直，并且不能自动更新
```bash
# 且在之前的版本
sudo apt-get install nvidia-fabricmanager-520

driver_version=520.61. 05
driver_version_main=$(echo $driver_version | awk -F '.' '{print $1}')
# wget http://mirrors.cloud.aliyuncs.com/nvidia-cuda/ubuntu2004/x86_64/nvidia-fabricmanager-${driver_version_main}_${driver_version}-1_amd64.deb
wget https://developer.download.nvidia.cn/compute/cuda/repos/ubuntu2004/x86_64/nvidia-fabricmanager-${driver_version_main}_${driver_version}-1_amd64.deb

dpkg -i nvidia-fabricmanager-${driver_version_main}_${driver_version}-1_amd64.deb

sudo systemctl enable nvidia-fabricmanager.service
sudo service nvidia-fabricmanager start

systemctl status nvidia-fabricmanager
```

# 安装后配置系统环境变量
```
vi /etc/profile

export CUDA_HOME=/usr/local/cuda
export PATH=$PATH:$CUDA_HOME/bin

export LD_LIBRARY_PATH=/usr/lib64:$CUDA_HOME/lib64
```