
#  [视频教程 快速入门](https://www.bilibili.com/video/BV1X84y1y7xy/?vd_source=bddb004da42430029e7bd52d0bdd0fe7)



# 注意： 
 - 将tme-dev分支 fork到自己的仓库，不然没法发起pr，成为贡献者
 - 包含info.json文件的应用为已整理的AIHUB应用，可以跳过

# 模型列表/分工/进度

https://docs.qq.com/sheet/DT0tPcWxHTG9OWUZE?tab=BB08J2

[代码中的常用方法](https://docs.qq.com/doc/DUkZoWUZ6bUxwUXl3)

# 应用文件结构

其中（内容基本已自动填写）
 - Dockerfile为镜像构建
 - init.sh位初始化脚本
 - app.py为应用启动(训练/推理/服务)，需要补齐Model类的基础参数
 - 其他自行添加配套内容

镜像调试，基础镜像为conda环境。先使用如下命令启动基础环境进入容器

```bash
# 进入模型应用
# 获取当前项目名作为应用名
aiapp=$(basename `pwd`)
cube_dir=($(dirname $(dirname "$PWD")))
chmod +x $cube_dir/src/docker/entrypoint.sh
sudo docker run --name ${aiapp} --privileged -it -e APPNAME=$aiapp -v $cube_dir/src:/src -v $PWD:/app -p 80:80 --entrypoint='/src/docker/entrypoint.sh' ccr.ccs.tencentyun.com/cube-studio/modelscope:base-cuda11.3-python3.7 bash 

```

补全init.sh环境脚本，没有环境问题可以忽略。
```bash
# init.sh 脚本会被复制到容器/根目录下，下载的环境文件不要放置在容器/app/目录下，不然会被加载到git
cp init.sh /init.sh && bash /init.sh
```
补齐app.py，运行调试，参考app1/app.py
```bash
/src/docker/entrypoint.sh python app.py
```

# 图像处理技巧

图上处理都有些技巧。  
1、图片resize 输入输出图像，避免输入输出过大，  
2、尽可能不经过中间磁盘储存，不然视频流进来的话跟不上推理速度。    
3、对于不同用户结果要把处理结果尽量添加随机数，不然不同请求结果可能存储冲突，  
4、中间文件都在result目录下面，这个目录gitignore了，不然会被git加载太大了。   
5、比较大的必须文件放在init脚本里面download下来，不要放在代码目录下，不然会被git加载  
6、如果涉及到其他的外部项目，可以下载到镜像/github下面，如果必须放在当前目录，建议软链过来  


# 记录模型效果

在模型app.py文件末尾添加注释，描述下列内容：

模型大小：  
模型效果：  
推理性能：  
占用内存/gpu：  
巧妙使用方法：  

# 用户：部署体验应用
首先需要部署docker
```bash
# 获取当前项目名作为应用名
aiapp=$(basename `pwd`)
cube_dir=($(dirname $(dirname "$PWD")))
chmod +x $cube_dir/src/docker/entrypoint.sh
sudo docker run --name ${aiapp} --privileged -d -e APPNAME=$aiapp -v $cube_dir/src:/src -v $PWD:/app -p 80:80 --entrypoint='/src/docker/entrypoint.sh' ccr.ccs.tencentyun.com/cube-studio/modelscope:base-cuda11.3-python3.7 sh /app/init.sh && python app.py 

```


# 常用方法汇总

图上处理都有些技巧：

 - 1、图片resize 输入输出图像，避免输入输出过大，
 - 2、尽可能不经过中间磁盘储存，不然视频流进来的话跟不上推理速度。
 - 3、对于不同用户结果要把处理结果尽量添加随机数，不然不同请求结果可能存储冲突，
 - 4、中间文件都在result目录下面，这个目录gitignore了，不然会被git加载太大了。
 - 5、比较大的必须文件放在init脚本里面download下来，不要放在代码目录下，不然会被git加载
 - 6、如果涉及到其他的外部项目，可以下载到镜像/github下面，如果必须放在当前目录，建议软链过来

# 代码技巧：
 - 1.删除代码中无用的 print 。
 - 2.去除没有引用的包提高性能 。
 - 3.在返回结果时候只返回有结果的部分，无结果部分去除 。

# 部分常用代码

1.输入与输出的图片，按比例缩小到最大边不大于1280。
```python
import cv2

def resize_image(image):
    height, width = image.shape[:2]
    max_size = 1280
    if max(height, width) > max_size:
        if height > width:
            ratio = max_size / height
        else:
            ratio = max_size / width
        image = cv2.resize(image, (int(width * ratio), int(height * ratio)))
    return image
```


2.结果中存在数组形式数据未显示在图片上，可将检测到的结果位置通过以下代码绘制到图片上。
```bash
import cv2

def box_label(self, input_path, res):
    def get_color(idx):
        idx = (idx + 1) * 3
        color = ((10 * idx) % 255, (20 * idx) % 255, (30 * idx) % 255)
        return color
    img = cv2.imread(input_path)
    unique_label = list(set(res['labels']))
    for idx in range(len(res['scores'])):
        x1, y1, x2, y2 = res['boxes'][idx]
        score = str("%.2f"%res['scores'][idx])
        label = str(res['labels'][idx])
        color = get_color(unique_label.index(label))
        line_width = int(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
        text_size = 0.001 * (img.shape[0] + img.shape[1]) / 2 + 0.3
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, line_width)
        cv2.putText(img, label, (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_PLAIN, text_size, color, line_width)
        cv2.putText(img, score, (int(x1), int(y2) + 10 + int(text_size*5)),
                    cv2.FONT_HERSHEY_PLAIN, text_size, color, line_width)
    return img
```


3.若遇到了需要自己去修改图像的场景，可用以下方法修改：

1）画框
```python
cv2.rectangle(img, pt1, pt2, color, thickness, lineType, shift )
参数表示依次为： （图片，长方形框左上角坐标, 长方形框右下角坐标， 字体颜色，字体粗细）
使用方法：
import cv2

cv2.rectangle(frame,                         # 加载的图片
              (int(bbox[0]), int(bbox[1])),  # 左上角坐标
              (int(bbox[2]), int(bbox[3])),  # 右上角坐标
               color,                        # 颜色
               2)                            # 粗细

```

 2）画圆
```python
	cv2.circle(img, center, radius, color, thickness=None, lineType=None, shift=None): 
	参数表示依次为：图片，中心点，半径，颜色，圆轮廓粗细，在中心坐标和半径值中的小数位数。
       使用方法：
import cv2

cv2.circle(img,      # 图像
           (447,63), # 中心点位置
           63,       # 半径
           (0,0,255),# 颜色 
           -1)       # 粗细  -1表示填充

```

 3）写字
```bash
	cv2.putText(image, text, org, font, fontScale, color, thickness, lineType, bottomLeftOrigin)
参数表示依次为：图片，要添加的文字，文字添加到图片上的位置，字体的类型，字体大小，字体颜色，字体粗细
使用方法：
import cv2

cv2.putText(img,                # 图像
            'OpenCV',           # 添加的文字
            (10, 500),          # 位置
            font,               # 字体
            4,                  # 字体大小
            (255, 255, 255),    # 颜色
            2)                  # 粗细

	
```


