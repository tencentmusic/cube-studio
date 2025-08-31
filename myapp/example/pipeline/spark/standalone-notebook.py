
# pyspark的版本要和服务端对应
# pip install pyspark==3.4.1 --index-url https://mirrors.aliyun.com/pypi/simple
import os
import sys
from random import random
from operator import add
import sys
from pyspark.sql import SparkSession

if __name__ == "__main__":
    """
        Usage: pi [partitions]
    """
    # 创建 SparkSession
    spark = SparkSession.builder \
        .appName("PythonPi") \
        .master('spark://myspark-master-svc.kubeflow:7077') \
        .config("spark.executor.memory", "2g") \
        .config("spark.executor.cores", "2") \
        .config("spark.driver.memory", "2g") \
        .config("spark.ui.enabled", False) \
        .config("spark.driver.port", os.getenv('PORT1')) \
        .config("spark.blockManager.port", os.getenv('PORT2')) \
        .config("spark.driver.bindAddress", '0.0.0.0') \
        .config("spark.driver.host", os.getenv('SERVICE_EXTERNAL_IP')) \
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