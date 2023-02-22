# coding: utf-8
import os,shutil
import pysnooper
from setuptools import setup, find_packages
from setuptools.command.install import install

class PostInstallCommand(install):
    def run(self):
        # 这里加入我们要执行的代码
        # 更新nginx配置文件
        base_dir = os.path.dirname(os.path.abspath(__file__))
        nginx_path=os.path.join(base_dir,'src/cubestudio/aihub/docker/default.conf')
        if os.path.exists(nginx_path):
            shutil.copyfile(nginx_path,'/etc/nginx/conf.d/default.conf')
        # 更新entrypoint
        entrypoint_path = os.path.join(base_dir,'src/cubestudio/aihub/docker/entrypoint.sh')
        if os.path.exists(entrypoint_path):
            shutil.copyfile(entrypoint_path, '/entrypoint.sh')
        install.run(self)

setup(
    name='cubestudio',  # 包名
    entry_points = {
        'console_scripts': [
            'cube=cubestudio.cli.tool:main'
        ],
    },
    cmdclass={
        'install': PostInstallCommand,
    },
    version='2022.10.1',  # 版本号
    description='模型市场，数据集市场',
    long_description='',
    author='tencent',
    author_email='pengluan@tencent.com',
    url='https://github.com/tencentmusic/cube-studio',
    license='',
    install_requires=[
        'PySnooper==1.1.1',
        'requests==2.28.1',
        'Flask==2.2.2',
        'kubernetes==26.1.0',
        'cryptography==39.0.1',
        'tqdm==4.64.1',
        'pyarrow==11.0.0',
        'celery==5.2.7',
        'redis==4.5.1',
        'fsspec==2023.1.0',
        'aiohttp==3.8.4',
        'librosa==0.10.0',
        'pandarallel==1.6.4',
        'requests_toolbelt==0.10.1',
        'multiprocess==0.70.14'
    ],
    python_requires='>=3.7',
    keywords='',
    packages=find_packages("src"),  # 必填 包含所有的py文件
    package_dir={'': 'src'},  # 必填 包的地址
    include_package_data=True,  # 将数据文件也打包
    )
