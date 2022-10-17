# coding: utf-8

from setuptools import setup, find_packages

setup(name='cubestudio',  # 包名
      version='2022.10.1',  # 版本号
      description='',
      long_description='',
      author='tencentmusic',
      author_email='pengluan@tencent.com',
      url='https://github.com/tencentmusic/cube-studio',
      license='',
      install_requires=[
          'pysnooper',
          'pika',
          'requests',
          'flask',
          'podman',
          'kubernetes'
      ],
      python_requires='>=3.6',
      keywords='',
      packages=find_packages('src'),  # 必填 包含所有的py文件
      package_dir={'': 'src'},  # 必填 包的地址
      include_package_data=True,  # 将数据文件也打包
      )
