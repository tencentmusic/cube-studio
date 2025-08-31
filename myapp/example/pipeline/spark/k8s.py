

import sys
from random import random
from operator import add

from pyspark.sql import SparkSession


if __name__ == "__main__":
    """
        Usage: pi [partitions]
    """
    # 创建 SparkSession
    spark = SparkSession.builder \
        .appName("PythonPi") \
        .master("k8s://https://kubernetes.default:443") \
        .config("spark.kubernetes.container.image", "ccr.ccs.tencentyun.com/cube-studio/spark-operator:spark-v3.1.1") \
        .config("spark.executor.instances", "2") \
        .config("spark.kubernetes.namespace", "pipeline") \
        .config("spark.kubernetes.authenticate.driver.serviceAccountName", "kubeflow-pipeline") \
        .getOrCreate()

    partitions = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    n = 100 * partitions

    def f(_):
        x = random() * 2 - 1
        y = random() * 2 - 1
        return 1 if x ** 2 + y ** 2 <= 1 else 0

    count = spark.sparkContext.parallelize(range(1, n + 1), partitions).map(f).reduce(add)
    print("Pi is roughly %f" % (4.0 * count / n))

    spark.stop()