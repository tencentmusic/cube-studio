
import requests,os,pysnooper,math
from requests_toolbelt import MultipartEncoder

# @pysnooper.snoop()
def download_file(url,des_dir=None,local_path=None):
    if des_dir:
        local_path = os.path.join(des_dir, url.split('/')[-1])
    print(f'begin donwload {local_path} from {url}')
    # 注意传入参数 stream=True
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        if os.path.exists(local_path):
            print(local_path,'已经存在')
            return
        with open(local_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=102400):
                f.write(chunk)
        r.close()

import hashlib


def get_md5(path):
    m = hashlib.md5()
    with open(path, 'rb') as f:
        for line in f:
            m.update(line)
    md5code = m.hexdigest()
    return md5code

# @pysnooper.snoop()
def upload_file(file_path,url,**kwargs):
    chunk_size = 1024 * 1024 * 2
    filename = os.path.basename(file_path)
    total_size = os.path.getsize(file_path)
    current_chunk = 0
    total_chunk = math.ceil(total_size / chunk_size)

    while current_chunk < total_chunk:
        start = current_chunk * chunk_size
        end = min(total_size, start + chunk_size)
        with open(file_path, 'rb') as f:
            f.seek(start)
            file_chunk_data = f.read(end - start)
        data = MultipartEncoder(
            fields={
                "filename": filename,
                "total_size": str(total_size),
                "current_chunk": str(current_chunk),
                "current_offset":str(start),
                "total_chunk": str(total_chunk),
                # "md5": get_md5(file_path),
                "file": (filename,file_chunk_data)
            }
        )
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Content-Type": data.content_type
        }
        headers.update(kwargs.get('headers',{}))
        with requests.post(url, headers=headers, data=data) as response:
            assert response.status_code == 200

        current_chunk = current_chunk + 1


# todo 压缩目录
def compress_dir(dir_path, compressed_file_path):
    pass


# todo 解压缩目录
# @pysnooper.snoop()
def decompress_file(compressed_file_path,des_dir):
    if '.zip' in compressed_file_path:
        import zipfile
        zFile = zipfile.ZipFile(compressed_file_path, "r")
        for fileM in zFile.namelist():
            zFile.extract(fileM, des_dir)
        zFile.close()

    if '.tar.gz' in compressed_file_path:
        import tarfile
        tf = tarfile.open(compressed_file_path)
        tf.extractall(des_dir)

    pass
