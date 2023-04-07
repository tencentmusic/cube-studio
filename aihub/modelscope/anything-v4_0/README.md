

感谢[Linaqruf](https://huggingface.co/Linaqruf)提供的model card供参考。

# Anything V4

欢迎使用Anything V4 - 一个为日本动漫爱好者设计的latent diffusion模型。这个模型旨在仅用少数几个提示词就能生成高质量、高细节的动漫风格。与其他动漫风格的Stable Diffusion模型一样，它还支持danbooru标签来生成图像。


例如： **_1girl, white hair, golden eyes, beautiful eyes, detail, flower meadow, cumulonimbus clouds, lighting, detailed sky, garden_** 

我感觉V4.5版本效果更好。该模型也在这个repo中，你可以尝试一下。


## 使用方式


```python
from modelscope.utils.constant import Tasks
from modelscope.pipelines import pipeline
import cv2

pipe = pipeline(task=Tasks.text_to_image_synthesis,
                model='dienstag/anything-v4.0',
                model_revision='v4.0')

prompt = 'masterpiece, best quality, 1girl, white hair, medium hair, cat ears, closed eyes, looking at viewer, :3, cute, scarf, jacket, outdoors, streets'
output = pipe({'text': prompt})
cv2.imwrite('result.png', output['output_imgs'][0])
```

## 例子

以下是该模型生成的一些样例
**Anime Girl:**
![Anime Girl](example-1.png)
```
masterpiece, best quality, 1girl, white hair, medium hair, cat ears, closed eyes, looking at viewer, :3, cute, scarf, jacket, outdoors, streets
Steps: 20, Sampler: DPM++ 2M Karras, CFG scale: 7
```
**Anime Boy:**
![Anime Boy](example-2.png)
```
1boy, bishounen, casual, indoors, sitting, coffee shop, bokeh
Steps: 20, Sampler: DPM++ 2M Karras, CFG scale: 7
```
**Scenery:**
![Scenery](example-4.png)
```
scenery, village, outdoors, sky, clouds
Steps: 50, Sampler: DPM++ 2S a Karras, CFG scale: 7
```

## License

这个模型是开放的，所有人都可以使用，CreativeML OpenRAIL-M许可证进一步规定了权利和使用。CreativeML OpenRAIL许可证规定了：

1. 你不能使用这个模型来故意产生或分享非法或有害的输出或内容。
2. 作者对你产生的输出结果没有任何权利，你可以自由使用它们，并对它们的使用负责，但不得违反许可证中的规定。
3. 你可以重新发布模型权重，并在商业上作为一项服务使用该模型。如果你这样做，请注意你必须包括与许可证中相同的使用限制，并将CreativeML OpenRAIL-M的副本分享给你的所有用户（请完全仔细阅读许可证）。 [请在这里阅读完整的许可证](https://huggingface.co/spaces/CompVis/stable-diffusion-license)。



## 郑重感谢

- [Linaqruf](https://huggingface.co/Linaqruf). [NoCrypt](https://huggingface.co/NoCrypt), 和 Fannovel16#9022 在模型训练上给予我的帮助。

