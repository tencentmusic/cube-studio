今天再给大家分享一个稍微复杂些的项目——给漫画上色。

<img width="300" alt="640" src="https://user-images.githubusercontent.com/20157705/191451414-772d01b3-e198-4f35-85cc-9f2f02c94236.png">


这类项目一般是通过GAN（生成对抗网络）来实现的，GAN一般有两个基础的组件，一个是生成器，另一个是判别器。通俗地理解，生成器生成的目标图片想尽可能地骗过判别器，判别器则要擦亮双眼尽量找出其中的破绽，通过这两个组件不断对抗，最终生成器生成的图片可以达到以假乱真的目的。

该项目有以下几个特性：

 - 自动跳过彩色图片 将彩色图片复制到(或放大到)输出文件夹。
 - 为小显存 GPU 添加图片分块 选项。
 - 添加超分辨率 Real-ESRGAN（支持 分块）默认输出75 webp减少体积。
 - 单独的上色、超分辨率等模式。

使用方法如下。

首先下载训练好的生成器模型文件放到./networks目录中

```bash
wget https://cubeatic.com/index.php/s/PcB4WgBnHXEKJrE/download -O generator.pt
```
然后，执行python inference.py即可。执行该命令，可附带一些参数来控制模型的输出，下面列了几个核心的参数：

```bash
-g：使用 GPU
-onlysr：仅放大模式(无上色)
-ca：强制上色模式
-nosr：仅上色模式(无放大)
-sub：处理输入路径（包括子文件夹下）的所有文件
```
项目地址：https://github.com/FlotingDream/Manga-Colorization-FJ/blob/main/README_CN.md

模型定义在项目./networks/models.py源文件中可以查看。模型还是比较复杂的，源代码只提供了生成器的网络结构。

想自己训练模型的朋友，可以参考Real-ESRGAN项目，也是在GitHub上开源的项目

