项目地址：https://github.com/AliaksandrSiarohin/first-order-model


# 示例动画

左侧的视频显示了驾驶视频。每个数据集右侧的第一行显示源视频。底行包含动画序列，其中运动从驾驶视频和从源图像中获取的对象传输。我们为每个任务训练了一个单独的网络。

VoxCeleb 数据集

[vox](https://github.com/AliaksandrSiarohin/first-order-model/blob/master/sup-mat/vox-teaser.gif)

时尚数据集

[fashion](https://github.com/AliaksandrSiarohin/first-order-model/raw/master/sup-mat/fashion-teaser.gif)

MGIF 数据集

[mgif](https://github.com/AliaksandrSiarohin/first-order-model/raw/master/sup-mat/mgif-teaser.gif)

# 安装

支持python3。要安装依赖项，请运行：

```bash
pip install -r requirements.txt
```
# YAML 配置

有几个配置 ( config/dataset_name.yaml) 文件，每个文件一个dataset。请参阅config/taichi-256.yaml以获取每个参数的描述。

# 预训练的检查点
检查点可以在以下链接中找到：google-drive或yandex-disk。

# 动画演示
要运行演示，请下载检查点并运行以下命令：

`python demo.py  --config config/dataset_name.yaml --driving_video path/to/driving --source_image path/to/source --checkpoint path/to/checkpoint --relative --adapt_scale`
结果将存储在result.mp4.

在我们的方法中使用之前，应该裁剪驾驶视频和源图像。要获得一些半自动裁剪建议，您可以使用python crop-video.py --inp some_youtube_video.mp4. 它将使用 ffmpeg 为作物生成命令。为了使用该脚本，需要 face-alligment 库：

```bash
git clone https://github.com/1adrianb/face-alignment
cd face-alignment
pip install -r requirements.txt
python setup.py install
```
