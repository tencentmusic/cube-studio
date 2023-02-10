

# 每个基础数据的类型，每列的类型，也就是所谓的features： int，double，string，enum，list,image，audio，video，Label等类型，在元数据信息中，媒体文件image/audio/video为本地地址或url地址
class FeatureDataType():
    # todo 音频文件 soundfile读取为 'numpy.ndarray'
    # todo 图片文件 PIL 读取为 ImageFile
    pass

class ValueDataType(FeatureDataType):
    dtype="int|double|string|enum"
    choice = ["xx","xx"]
    value=''

class listDataType(FeatureDataType):
    dtype="int|double|str|enum"
    choice = ["xx","xx"]
    value = []

class ImageDataType(FeatureDataType):
    # img = PIL
    path = ''

class AudioDataType(FeatureDataType):
    array=[]
    path = ''
    sampling_rate = ''


