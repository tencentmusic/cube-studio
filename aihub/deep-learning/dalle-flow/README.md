今天给大家分享一个有趣的 AI 项目 —— dalle-flow。

该项目可以根据文本生成图片，GitHub 上已经开源。

项目地址：https://github.com/jina-ai/dalle-flow

下面演示下项目效果，并介绍用到的算法。


# 1. 效果演示

以一个简单的例子给大家演示一下。

比如：我们想为 a teddy bear on a skateboard in Times Square(在时代广场玩滑板的泰迪熊) 这段文本生成一张图片。

将它输入dalle-flow 后， 便可以得到下面的图片


<img width="400" alt="640" src="https://user-images.githubusercontent.com/20157705/191516749-93e81d17-7673-4165-8575-4fd2c453fc04.png">


是不是很神奇！

下面我用几行 Python 代码教大家使用这个项目。

首先，安装 docarray

```
pip install "docarray[common]>=0.13.5" jina
```
定义server_url变量，存放dalle-flow模型地址

```bash
server_url = 'grpc://dalle-flow.jina.ai:51005'
```

server_url是官方提供的服务，我们也可以按照文档，将模型部署到自己的服务器（需要有GPU）。

将文本提交到服务器，获得候选图片。

```bash
prompt = 'a teddy bear on a skateboard in Times Square'
from docarray import Document

da = Document(text=prompt).post(server_url, parameters={'num_images': 2}).matches
```
提交文本后，服务器会调用DALL·E-Mega算法生成候选图像，然后调用CLIP-as-service 对候选图像进行排名。

我们指定num_images等于 2，最终会返回 4 张图片，2 张来自DALLE-mega模型，2 张来自GLID3 XL模型。由于server_url服务器在国外，程序运行时间可能会比较长，大家运行的时候要多等等。

程序运行结束后，我们将这 4 张图片展示出来

```bash
da.plot_image_sprites(fig_size=(10,10), show_index=True)
```

<img width="400" alt="640" src="https://user-images.githubusercontent.com/20157705/191516913-3d5133b4-4f42-42ab-8345-10c383ab7525.png">


我们可以选择其中一张，继续提交到服务器上进行diffusion。

每张图左上角都有一个编号，这里我选的是编号为 2 的图片

```bash
fav_id = 2
fav = da[fav_id]

diffused = fav.post(f'{server_url}', parameters={'skip_rate': 0.5, 'num_images': 36}, target_executor='diffusion').matches

```
diffusion其实是将选中的图片，送入GLID-3 XL模型，丰富纹理和背景。

返回结果如下：

<img width="400" alt="640" src="https://user-images.githubusercontent.com/20157705/191516997-761c0c3c-bf59-47e2-833a-be0c976c70bf.png">


我们可以从中选一张满意的图片作为最终的结果页。

```bash
fav = diffused[6]
fav.display()
```

# 2. 算法小知识

dalle-flow项目使用起来虽然很简单，但涉及的DALL·E算法却很复杂，这里只简单介绍下。

DALL·E的目标是把文本token和图像token当成一个数据序列，通过Transformer进行自回归。

<img width="800" alt="640" src="https://user-images.githubusercontent.com/20157705/191517080-7878c356-57da-41aa-be20-22210b4421ab.png">


这个过程跟机器翻译有些像，机器翻译是将英文文本翻译成中文文本，而DALL·E将英文文本翻译成图片，文本中的token是单词，而图像中的token则是像素。

对dalle-flow项目感兴趣的朋友可以自己跑跑上面的代码，自己部署模型试试。
