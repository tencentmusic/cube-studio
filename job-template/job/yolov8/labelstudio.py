import requests
import pysnooper
import os,time,json
import pysnooper

class LabelStudio_ML_Backend():

    # @pysnooper.snoop(watch_explode='kwargs')
    def labelstudio_health(self,  **kwargs):
        return {}

    # 标注平台发来本项目标注任务的元数据。看看是不是可以和当前模型配置
    # @pysnooper.snoop(watch_explode='kwargs')
    def labelstudio_setup(self,  **kwargs):
        self.access_token=kwargs['access_token']
        self.hostname = kwargs['hostname']
        self.project = kwargs['project']
        return {}


    # @pysnooper.snoop()
    def labelstudio_download_image(self,image_path,save_dir='result',**kwargs):
        headers = {
            "Authorization": f"Token {self.access_token}"
        }
        save_path = os.path.join(save_dir,f'result{int(time.time() * 1000)}.jpg')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if os.path.exists(save_path):
            os.remove(save_path)
        if 'http://' in image_path or "https://" in image_path:
            if self.hostname in image_path:
                import requests
                from urllib import parse
                from urllib.parse import urlparse
                res = urlparse(image_path)
                params = parse.parse_qs(res.query)
                d = params.get('d',[''])[0]
                image_path=f'http://{res.netloc}/static/'+d
                file = open(save_path,mode='wb')
                print(image_path,headers)
                # res = requests.get(image_path,headers=headers)
                res = requests.get(image_path)
                if res.status_code==200:
                    file.write(res.content)
                    file.close()
                    image_path=save_path
            else:
                import requests
                # 发送请求并获取图片内容
                response = requests.get(image_path)
                # 确保请求成功
                if response.status_code == 200:
                    # 将视频内容写入本地文件
                    with open(save_path, "wb") as file:
                        file.write(response.content)
                        print(f"图片已成功保存到: {save_path}")
                else:
                    print(f"请求失败，状态码: {response.status_code}")
                image_path = save_path

        return image_path

    # @pysnooper.snoop()
    def labelstudio_download_audio(self,audio_path,save_dir='result',**kwargs):
        headers = {
            "Authorization": f"Token {self.access_token}"
        }
        audio_format = audio_path.split(".")[-1]
        if len(audio_format)>5:
            audio_format='wav'
        save_path = os.path.join(save_dir,f'result{int(time.time() * 1000)}.{audio_format}')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if os.path.exists(save_path):
            os.remove(save_path)
        if 'http://' in audio_path or "https://" in audio_path:
            if self.hostname in audio_path:
                import requests
                from urllib import parse
                from urllib.parse import urlparse
                res = urlparse(audio_path)
                params = parse.parse_qs(res.query)
                d = params.get('d',[''])[0]
                audio_path=f'http://{res.netloc}/static/'+d
                file = open(save_path,mode='wb')
                print(audio_path,headers)
                # res = requests.get(image_path,headers=headers)
                res = requests.get(audio_path)
                if res.status_code==200:
                    file.write(res.content)
                    file.close()
                    audio_path=save_path
            else:
                import requests
                # 发送请求并获取音频内容
                response = requests.get(audio_path)
                # 确保请求成功
                if response.status_code == 200:
                    # 将视频内容写入本地文件
                    with open(save_path, "wb") as file:
                        file.write(response.content)
                        print(f"音频已成功保存到: {save_path}")
                else:
                    print(f"请求失败，状态码: {response.status_code}")
                audio_path = save_path

        return audio_path
