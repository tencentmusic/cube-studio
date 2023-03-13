import re
import shutil
from jinja2 import Environment, BaseLoader, DebugUndefined
import requests,json
import os,sys


# 下载所有模型的基础信息
def download_all():
    req={
        "PageNumber": 1,
        "PageSize": 1000,
        "SingleCriterion": [],
        "SortBy": "DownloadsCount",
        "Target": ""
    }

    res = requests.put('https://modelscope.cn/api/v1/dolphin/models',json=req)
    all_models = res.json().get("Data",{}).get("Model",{}).get('Models',[])
    json.dump(all_models,open("all_models.json",mode='w'),ensure_ascii=False,indent=4)
# download_all()
# exit(0)
def download_one():
    all_models = json.load(open("all_models.json",mode='r'))
    for model in all_models:

        model_name = model.get('Path','')+'/'+model.get("Name",'')
        save_path = os.path.join('modelscope',model_name+".json")
        print(save_path)
        os.makedirs(os.path.dirname(save_path),exist_ok=True)
        try:
            if not os.path.exists(save_path):
                url = f'https://modelscope.cn/api/v1/models/{model_name}'
                print(url)
                res = requests.get(url)
                result = res.json()
                json.dump(result,open(save_path,mode='w'),ensure_ascii=False,indent=4)
        except Exception as e:
            print(e)
        try:
            quickstart_save_path = os.path.join('modelscope', model_name + ".quickstart.json")
            if not os.path.exists(quickstart_save_path):
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

        # # 还在集成的不需要
        # if model['Integrating']<2:
        #     print(model_path,'集成中')
        #     continue
        # 使用有障碍的不要
        # if model['IsAccessible']==0:
        #     print(model_path, '有障碍')
        #     continue
        # 下载量太少的不要
        # if int(model['Downloads'])<100:
        #     print(model_path, '下载少')
        #     continue

        save_path = os.path.join('modelscope', model_path + ".json")
        model = json.load(open(save_path)).get("Data", {})

        aihub_save_dir = os.path.join('aihub', model_path.replace('_',"-").replace('.','_').lower())
        shutil.rmtree(aihub_save_dir,ignore_errors=True)
        # os.makedirs(aihub_save_dir, exist_ok=True)
        shutil.copytree('demo',aihub_save_dir)

        save_model_name = model_name.replace('_','-').lower()
        # readme
        readme_doc = model.get('ReadMeContent','')
        path = os.path.join(aihub_save_dir,'README.md')
        file = open(path,mode='w')
        file.write(readme_doc)
        file.close()

        # Dockerfile
        path = os.path.join(aihub_save_dir, 'Dockerfile')
        content = open(path, mode='r').readlines()
        content = ''.join(content)
        rtemplate = Environment(loader=BaseLoader, undefined=DebugUndefined).from_string(content)
        des_str = rtemplate.render(APP_NAME=save_model_name)
        file = open(path, mode='w')
        file.write(des_str)
        file.close()


        app={
            "name":save_model_name,
            "label":model['ChineseName'],
            "describe":model['Description'],
            "hot":model['Downloads'],
            "frameworks":' '.join(model['Frameworks']),
            "scenes":'',
            "url":f"https://modelscope.cn/models/{model_path}/summary",
            "web_examples":[],
            "resource_gpu":"0",
            "inference_inputs":[],
            "train_inputs": [],
            "load_model_fun":"pass",
            "inference_fun":"pass",
            "inference_fun_args":"arg1",
            "train_fun":"pass"
        }

        field = ' '.join(model['Domain'])
        field = '自然语言' if 'nlp' in field else '机器视觉' if 'cv' in field else '听觉' if 'audio' in field else '多模态' if 'multi-modal' in field else "未知"
        app['field']=field

        model_cache_path = f'/root/.cache/modelscope/hub/{model_path}/'

        widgets = model['widgets']
        if widgets:
            # 从widgets中读取example
            examples = widgets[0]['examples']
            if examples:
                aihub_web_examples=[]
                for example in examples:
                    aihub_example = {
                        "label":example['title'],
                        "input":{}
                    }
                    for i,input in enumerate(example['inputs']):
                        name = input['name']
                        if not name:
                            name = f'arg{i}'
                        if type(input['data'])==str:
                            aihub_example['input'][name]= input['data'].replace("git://",model_cache_path)
                        else:
                            aihub_example['input'][name] = input['data']

                    aihub_web_examples.append(aihub_example)
                # print(aihub_web_examples)
                app['web_examples']=json.dumps(aihub_web_examples,indent=4,ensure_ascii=False).replace('\n','\n    ')

            # 从widgets中读取推理资源
            inferencespec = widgets[0].get('inferencespec',{})
            if inferencespec:
                app['resource_gpu']=inferencespec.get('gpu','0')

            # 从widgets中读取input
            inference_fun_args=[]
            inference_inputs = widgets[0].get('inputs',[])
            if inference_inputs:
                aihub_input=[]
                # print(model_path)
                for i,inference_input in enumerate(inference_inputs):
                    validator = inference_input['validator']   # 音视频 {'max_resolution': '4096*4096', 'max_size': '10M'}   文本 max_words
                    aihub_validators=None
                    if validator and 'max_words' in validator:
                        aihub_validators=f"Validator(max={validator['max_words']})"
                    input_type = inference_input['type']
                    input_type = input_type.replace("-list",'_select')
                    name = inference_input["name"]
                    if not name:
                        name = f'arg{i}'
                    inference_fun_args.append(name)
                    # from cubestudio.aihub.model import Model, Validator, Field_type, Field
                    input = f'''Field(type=Field_type.{input_type}, name='{name}', label='{inference_input["title"]}',describe='{inference_input["title"]}',default='',validators={aihub_validators})'''
                    aihub_input.append(input)

                app['inference_inputs']=json.dumps(aihub_input,ensure_ascii=False,indent=4).replace('"','').replace('\n','\n    ')
                inference_fun_args=','.join(inference_fun_args)
                app['inference_fun_args'] = inference_fun_args
                # print(app['inference_inputs'])


        # 函数 先从快速启动中读取，再尝试从task中读取
        save_path = os.path.join('modelscope', model_path + ".quickstart.json")
        if os.path.exists(save_path):
            quickstart = json.load(open(save_path)).get("Data", {}).get("Quickstart",'')
            if '## 模型加载和推理' in quickstart:
                quickstart = quickstart[quickstart.index('## 模型加载和推理'):]
                python_code = re.findall('```python([\s\S]*?)```',quickstart)
                if len(python_code)>0:
                    load_model_fun = python_code[0].strip('\n')
                    load_model_fun = load_model_fun.split("\n")
                    load_model_fun[-1] = "self."+load_model_fun[-1]
                    load_model_fun='\n'.join(load_model_fun)
                    load_model_fun = load_model_fun.replace('\n', '\n        ')

                    app['load_model_fun']=load_model_fun
                if len(python_code) > 1:
                    inference_fun = python_code[1].strip('\n')
                    # print(inference_fun)
                    app['inference_fun'] = ("result = self."+inference_fun).replace('\n', '\n        ')

                # print(quickstart)
                # print(python_code)

        # # 从readme中尝试读取训练函数
        # if '##训练' in readme_doc:
        #     print(readme_doc)
        # # 从readme中尝试读取依赖安装


        # download_model
        path = os.path.join(aihub_save_dir, 'download_model.py')
        content = open(path, mode='r').readlines()
        content = ''.join(content)
        rtemplate = Environment(loader=BaseLoader, undefined=DebugUndefined).from_string(content)
        download_fun = app['load_model_fun']
        download_fun = download_fun.replace('        ','').replace('self.','')
        if not download_fun:
            download_fun='''
from modelscope.hub.snapshot_download import snapshot_download
model_dir = snapshot_download('%s', cache_dir='/root/.cache/modelscope/hub/')
            '''%model_path
        des_str = rtemplate.render(download_fun=download_fun)
        print(des_str)
        print(path)
        file = open(path, mode='w')
        file.write(des_str)
        file.close()


        # app
        path = os.path.join(aihub_save_dir, 'app.py')
        content = open(path, mode='r').readlines()
        content = ''.join(content)
        rtemplate = Environment(loader=BaseLoader, undefined=DebugUndefined).from_string(content)
        # print(model_path)

        from collections import namedtuple
        app = namedtuple('Struct', app.keys())(*app.values())
        des_str = rtemplate.render(app=app)
        file = open(path, mode='w')
        file.write(des_str)
        file.close()



make_aihub()