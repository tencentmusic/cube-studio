import ray,os
import pandas
import numpy

@ray.remote
def generate_features(file_path,i):
    print(file_path,i)
    data = pandas.read_csv(file_path)
    # 将数据分割成多个部分，每个部分将在一个不同的worker上进行处理
    data_splits = numpy.array_split(data, 10)
    # 在这里执行你的特征工程代码
    features = data_splits[i].copy()
    # features['new_feature'] = data['column1'] + data['column2']
    features['new_feature'] = data['age'] + data['duration']
    features=features[['age','duration','new_feature']]
    return features


def main():
    # 读取数据
    file_path='/mnt/admin/pipeline/example/feature-process/data-test1.csv'

    # 使用Ray并行生成特征
    features = ray.get([generate_features.remote(file_path,i) for i in range(0,10)])

    # 合并所有生成的特征
    final_features = pandas.concat(features)
    print(final_features)
    final_features.to_csv('/mnt/admin/pipeline/example/feature-process/result.csv',index=False)


if __name__ == "__main__":

    head_service_ip = os.getenv('RAY_HOST', '')
    if head_service_ip:
        # 集群模式
        head_host = head_service_ip + ".pipeline" + ":10001"
        ray.util.connect(head_host)
    else:
        # 本地模式
        ray.init()

    main()