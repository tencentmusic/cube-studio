


# 规范标注类型，合成类型
# todo 合并标注结果，音频检测，声纹识别，音频识别，音频场景分类。
# todo 文本分词，文本翻译，文本语音合成，文本

class LabelType():
    name = 'label'
    label = '标注名称'

class CLASSIFICATION(LabelType):
    name = 'classification'
    label = '分类'
    attributes=[
        {
            "name": "name",
            "type": "str"
        }
    ]
    num_classes = 2


class DETECTION(LabelType):
    name = 'detection'
    label = '检测'
    attributes = [
        {
            "name": "left",
            "type": "double"
        },
        {
            "name": "top",
            "type": "double"
        },
        {
            "name": "width",
            "type": "double"
        },
        {
            "name": "height",
            "type": "double"
        }
    ]

    pass
