
import sys,inspect

# 找出模块里所有的类名
def get_classes(arg):
    all_class = {}
    clsmembers = inspect.getmembers(arg, inspect.isclass)
    for (name, class_) in clsmembers:
        all_class[name]=class_
    return all_class