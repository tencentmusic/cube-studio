
# 模型介绍
本模型是个性化语音合成的自动标注工具所依赖的模型及资源集合


## 框架描述
暂无

## 使用方式和范围

使用方式：
* 使用请确保modelscope已经更新到1.4.0版本及以上并通过以下命令安装tts-autolabel及相关依赖包:
> pip install tts-autolabel -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html

另外tts-autolabel包依赖的onnxruntime，默认onnxruntime安装版本大于等于1.11会导致要求numpy版本大于1.21，在Modelscope默认镜像中，因为默认安装tensorflow-1.15，该版本tensorflow要求numpy小于等于1.18，这样会出现冲突。碰到这种情况，可以参考如下方案解决：
1. 考虑如果使用场景不需要tensorflow，可以删除tensorflow-1.15
2. 考虑换onnxruntime==1.10版本，然后重新安装numpy

使用范围:
* 适用于中英文的语音合成数据自动标注场景，输入wav格式音频文件夹，音频文件命名规范为“数字id+.wav后缀”，建议一次性不要处理太多音频，暂时不支持断点续跑的功能。
* 目前仅支持Linux环境使用。

目标场景:
* 各种语音合成任务的数据标注辅助

### 如何使用
参考下方代码范例

#### 代码范例
如在Notebook下使用请先在代码块中安装`tts-autolabel依赖`，
```python
# 运行此代码块安装tts-autolabel
import sys
!{sys.executable} -m pip install tts-autolabel -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
```
调用接口运行自动标注，
```Python

from modelscope.tools import run_auto_label

input_wav = '/audio/path' # wav audio path
work_dir = '/output/path' # output path
ret, report = run_auto_label(input_wav = input_wav,
                             work_dir = work_dir)
print(report)
```

### 模型局限性以及可能的偏差
* **目前支持中文、中英混合(少量基础英文)音频的标注，英文及其他语种的支持将在后续版本中更新。**
* **仅支持Linux环境，基于gcc4.8.5，过低版本glibc可能会出现不兼容情况**

## 训练数据介绍
无

## 模型训练流程
无

### 预处理
无

## 数据评估及结果
暂无

## 引用
暂无

