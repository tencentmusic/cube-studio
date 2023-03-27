import re
import shutil
from jinja2 import Environment, BaseLoader, DebugUndefined
import requests,json
import os,sys

def download_one():
    all_models = json.load(open("all_models.json",mode='r'))
    for model in all_models:

        model_name = model.get('Path','')+'/'+model.get("Name",'')
        save_path = os.path.join('modelscope',model_name+".json")
        print(save_path)
        os.makedirs(os.path.dirname(save_path),exist_ok=True)
        try:
            url = f'https://modelscope.cn/api/v1/models/{model_name}'
            print(url)
            res = requests.get(url)
            result = res.json()
            json.dump(result,open(save_path,mode='w'),ensure_ascii=False,indent=4)
        except Exception as e:
            print(e)
        try:
            quickstart_save_path = os.path.join('modelscope', model_name + ".quickstart.json")
            url = f'https://modelscope.cn/api/v1/models/{model_name}/quickstart'
            print(url)
            res = requests.get(url)
            result = res.json()
            json.dump(result, open(quickstart_save_path, mode='w'), ensure_ascii=False, indent=4)
        except Exception as e:
            print(e)



# 下载每个模型的详细信息
# download_one()
# exit(1)
def make_aihub():
    all_models = json.load(open("all_models.json", mode='r'))
    for model in all_models:
        model_name = model.get("Name", '')
        model_path = model.get('Path', '') + '/' + model.get("Name", '')
        save_model_name = model_name.replace('_', '-').replace('.','_').lower()
        # 已经处理过的不再处理
        if os.path.exists(save_model_name):

            save_path = os.path.join('modelscope', model_path + ".json")
            model = json.load(open(save_path)).get("Data", {})

            # readme
            readme_doc = model.get('ReadMeContent','')
            # print(readme_doc)
            images = re.findall('(\(.*?.jpg\))',readme_doc)
            if images:
                for image in images:
                    if 'http' not in image:
                        image = image[image[:-1].rindex('('):]
                        new_image = image[1:-1]
                        readme_doc = readme_doc.replace(image,'(https://modelscope.cn/api/v1/models/%s/%s/repo?Revision=master&FilePath=%s&View=true)'%(model.get('Path', ''),model.get("Name", ''),new_image))
                        # print(save_model_name,image)

            images = re.findall('(".*?.jpg")',readme_doc)
            if images:
                for image in images:
                    if 'http' not in image:
                        image = image[image[:-1].rindex('"'):]
                        new_image = image[1:-1]
                        readme_doc = readme_doc.replace(image,'"https://modelscope.cn/api/v1/models/%s/%s/repo?Revision=master&FilePath=%s&View=true"'%(model.get('Path', ''),model.get("Name", ''),new_image))
                        # print(save_model_name,image)

            path = os.path.join(save_model_name,'README.md')
            file = open(path,mode='w')
            file.write(readme_doc)
            file.close()



make_aihub()