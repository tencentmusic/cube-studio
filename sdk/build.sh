pip3 install -U pip setuptools wheel twine packaging upload keyring keyrings.alt
# 代码检查
python3 setup.py check
# 打包
rm -rf dist build
rm -rf cubestudio*
rm -rf src/cubestudio.egg-info
#python3 setup.py sdist
python setup.py sdist bdist_wheel
# 上传
#twine upload --skip-existing ./dist/cubestudio-2022.10.1.tar.gz


