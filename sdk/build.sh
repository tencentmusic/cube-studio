pip3 install -U pip setuptools wheel twine packaging upload keyring keyrings.alt
# 代码检查
python3 setup.py check
# 打包
rm -rf dist
python3 setup.py sdist
# 上传
twine upload --skip-existing ./dist/cube-studio-2022.10.1.tar.gz


