NoGAN是一种新型GAN，它能花费最少的时间进行GAN训练。

<img width="300" alt="640" src="https://user-images.githubusercontent.com/20157705/191451414-772d01b3-e198-4f35-85cc-9f2f02c94236.png">

1. 准备工作
首先，用git clone命令下载源码
```bash
git clone https://github.com/jantic/DeOldify.git
```

进入项目根目录，安装Python依赖包
```bash
pip3 install -r requirements.txt
```

编写代码运行项目之前，需要下载预训练好的模型。项目提供了三个模型


区别如下：

 - ColorizeArtistic_gen.pth：在有趣的细节和活力方面实现了最高质量的图像着色效果，该模型在 UNet 上使用 resnet34 为主干，通过 NoGAN 进行了 5 次评论家预训练/GAN 循环重复训练

 - ColorizeStable_gen.pth：在风景和肖像方面取得了最佳效果，该模型在 UNet 上使用 resnet101 为主干，通过 NoGAN 进行了 3 次评论家预训练/GAN 循环重复训练

 - ColorizeVideo_gen.pth：针对流畅的视频进行了优化，它仅使用初始生成器/评论家预训练/GAN NoGAN 训练。由于追求流畅的速度，它的色彩比前两者少。

将下载好的模型文件放在项目根目录的models目录下即可。


2. 编写代码

在项目根目录同级目录下创建Python文件，编写代码加载刚刚下载好的模型文件。

```bash
from DeOldify.deoldify.generators import gen_inference_wide
from DeOldify.deoldify.filters import MasterFilter, ColorizerFilter

# 指定模型文件
learn = gen_inference_wide(root_folder=Path('./DeOldify'), weights_name='ColorizeVideo_gen')

# 加载模型
deoldfly_model = MasterFilter([ColorizerFilter(learn=learn)], render_factor=10)
```
root_folder指定项目根目录，weights_name指定接下来使用哪个模型为照片上色。

读取老照片，进行上色
```bash
import cv2
import numpy as np
from PIL import Image

img = cv2.imread('./images/origin.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
pil_img = Image.fromarray(img)

filtered_image = deoldfly_model.filter(
    pil_img, pil_img, render_factor=35, post_process=True
)

result_img = np.asarray(filtered_image)
result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
cv2.imwrite('deoldify.jpg', result_img)
```
用cv2读取老照片，并用PIL.Image模块将图片转换成模型输入所需要的格式，送入模型进行上色，完成后保存。

上述代码是我从项目源码中抽取的，可以看到，运行代码还是非常简单的。
