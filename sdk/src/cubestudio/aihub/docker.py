
import os
import pysnooper
dockerfile='''
FROM ccr.ccs.tencentyun.com/cube-studio/aihub:base
WORKDIR /app
# 拷贝文件
{copy_command}
# 安装基础环境
{init_command)

ENTRYPOINT ["python", "app.py"]

'''
class Docker():
    def __init__(self,images):
        self.images=images

    # 构建镜像
    @pysnooper.snoop()
    def build(self,init,files):
        files = list(set([init]+files))
        copy_command=''
        init_command=''
        for file in files:
            if os.path.exists(file):
                copy_command+='COPY %s /app/%s'%(file,file)
        if init:
            init_command='RUN bash /app/%s'%init

        f = open('Dockerfile',mode='w')
        f.write(dockerfile.format(copy_command=copy_command,init_command=init_command))
        f.close()
        os.system('docker build -t %s -f Dockerfile .' % self.images)
        return 'build success'

    # 推送
    def push(self,user, password):
        host = self.images[0:self.images.index('/')]
        os.system('docker login --username %s --password %s %s' % (user, password, host))
        os.system('docker push %s' % (self.images))
        return 'push success'