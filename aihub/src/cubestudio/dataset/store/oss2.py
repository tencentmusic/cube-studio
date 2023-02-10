
import os
import sys
import oss2
import time
import logging


class OSS2_client():
    def __init__(self,key_id,key_secret,internal_endpoint,public_endpoint,bucket_name,timeout=30,internal=False):
        self.key_id=key_id
        self.key_secret=key_secret
        self.internal_endpoint=internal_endpoint
        self.public_endpoint=public_endpoint
        self.bucket_name=bucket_name
        self.timeout=timeout
        self.auth = oss2.Auth(self.key_id, self.key_secret)
        if internal:
            self.endpoint = self.internal_endpoint
        else:
            self.endpoint = self.public_endpoint
        self.service = oss2.Service(self.auth, self.endpoint)
        self.bucket=oss2.Bucket(self.auth, self.endpoint, self.bucket_name, connect_timeout=self.timeout)

    # 获取bucket_list
    def bucket_list(self):
        return [b.name for b in oss2.BucketIterator(self.service)]

    # 创建该bucket
    def create_bucket(self):
        try:
            self.bucket.create_bucket(oss2.models.BUCKET_ACL_PRIVATE)
        except Exception as e:
            print(e)

    # 删除多个文件
    def batch_delete(self,object_list):
        if object_list:
            result = self.bucket.batch_delete_objects(object_list)
            print(result)
            # 打印成功删除的文件名。
            print('\n'.join(result.deleted_keys))


    # 下载图片
    def downloadfiles(self,remote_paths,lical_dir):

        if not os.path.exists(lical_dir):
            os.makedirs(lical_dir)
            logging.info("The floder {0} is not existed, will creat it".format(lical_dir))

        start_time = time.time()

        for tmp_file in remote_paths:
            if not self.bucket.object_exists(tmp_file):
                logging.info("File {0} is not on the OSS!".format(tmp_file))
            else:
                logging.info("Will download {0} !".format(tmp_file))

                tmp_time = time.time()
                # cut the file name
                filename = tmp_file[tmp_file.rfind("/") + 1: len(tmp_file)]
                localFilename = os.path.join(lical_dir, filename)
                oss2.resumable_download(
                    self.bucket,
                    tmp_file,
                    localFilename,
                    progress_callback=self.percentage)
                logging.info("\nFile {0} -> {1} downloads finished, cost {2} Sec.".format(
                    tmp_file, localFilename, time.time() - tmp_time))

        logging.info("All download tasks have finished!")
        logging.info("Cost {0} Sec.".format(time.time() - start_time))

    # 上传单个文件
    def uploadfile(self,local_file_path,remote_file_path):
        begin_time = time.time()
        try:
            # bucket.put_object_from_file(oss_filename, tmp_file)
            oss2.resumable_upload(bucket=self.bucket, key=remote_file_path, filename=local_file_path,
                                  progress_callback=self.percentage,headers={"x-oss-object-acl":"public-read"})
        except Exception as e:
            logging.info('upload to oss failed: %s' % e)
            return False
        else:
            logging.info("uploads finished, cost %s Sec."%(str(time.time() - begin_time)))
            return True

    # 上传图片，只上传文件夹下的文件
    def uploadfiles(self,local_dir,remote_dir):
        print('start uploading files')
        start_time = time.time()

        for lists in os.listdir(local_dir):
            tmp_file = os.path.join(local_dir, lists)
            if os.path.isdir(tmp_file):
                continue
            if not os.path.exists(tmp_file):
                logging.info("File {0} is not exists!".format(tmp_file))
            else:
                logging.info("Will upload %s to the oss"%(tmp_file,))
                # 获取文件名称
                filename = tmp_file[tmp_file.rfind("/") + 1: len(tmp_file)]
                #print(filename)
                oss_filename = os.path.join(remote_dir, filename)
                # print(oss_filename)
                self.uploadfile(tmp_file,oss_filename)

        logging.info("All upload tasks have finished!")
        logging.info("Cost {0} Sec.".format(time.time() - start_time))

    # 显示在bucket上的所有文件路径.prefix 表示目录
    def showFiles(self,prefix=''):
        path=[]
        logging.info("Show All Files:")
        for obj in oss2.ObjectIterator(self.bucket, delimiter='/',prefix=prefix):
            path.append(obj.key)
            if obj.is_prefix():   # 通过is_prefix方法判断obj是否为文件夹。 判断是否是公共前缀（目录）
                print('directory: ' + obj.key)
            else:  # 文件
                print('file: ' + obj.key)
        return path



    # 获取上传速率，断点续传时使用
    def percentage(self,consumed_bytes, total_bytes):
        if total_bytes:
            rate = int(100 * (float(consumed_bytes) / float(total_bytes)))
            # logging.info('\r{0}% '.format(rate))

            sys.stdout.flush()


def main(CONFIG):

    oss_client = OSS2_client(bucket_name='luanpeng')
    # bucket_list = oss_client.bucket_list()
    # print(bucket_list)
    oss_client.create_bucket()
    print('list files in bucket')
    paths=oss_client.showFiles(prefix='vesionbook-face-image/2019-02-25/10/')
    print(paths)
    # print('upload file to bucket')
    # oss_client.uploadFiles('/home/luanpeng/data/python/cloudai2/src/cronjob/file/vesionbook-face-image/2019-02-25/10','vesionbook-face-image/2019-02-25/10')
    print('delete bucket')
    oss_client.batch_delete(paths)




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f',
        '--file',
        dest='files',
        action='append',
        default=[],
        help='the file name you want to download!')

    parser.add_argument(
        "-l",
        "--listfiles",
        default=False,
        dest="listFiles",
        action="store_true",
        help="If been True, list the All the files on the oss !")

    parser.add_argument(
        "-o",
        "--outputPath",
        dest="outputPath",
        default="./",
        type=str,
        help="the floder we want to save the files!")

    parser.add_argument(
        "-i",
        "--internal",
        dest="internal",
        default=False,
        action="store_true",
        help="if you using the internal network of aliyun ECS !")

    parser.add_argument(
        "--upload",
        dest="upload",
        default=False,
        action="store_true",
        help="If been used, the mode will be select local files to upload!")

    parser.add_argument(
        "-p",
        "--prefix",
        dest="prefix",
        default="",
        type=str,
        help="the prefix add to the upload files!")

    CONFIG, unparsed = parser.parse_known_args()
    print(CONFIG)
    main(CONFIG)


# if __name__ == '__main__':
#     bucket = oss2.Bucket(oss2.Auth(KEY_ID, KEY_SECRET), public_endpoint, bucket_name, connect_timeout=timeout)
#
#     bucket.put_object_from_file('test.txt', 'test.txt')  # upload
#     #bucket.get_object_to_file('test.txt', 'local_test.txt')  # download

