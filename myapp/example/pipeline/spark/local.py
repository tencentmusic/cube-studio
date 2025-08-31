from random import random
from operator import add
from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession\
    .builder\
    .appName("PythonPi-Local")\
    .master("local")\
    .getOrCreate()

    n = 100 * 2

    def f(_):
        x = random() * 2 - 1
        y = random() * 2 - 1
        return 1 if x ** 2 + y ** 2 <= 1 else 0

    count = spark.sparkContext.parallelize(range(1, n + 1), 2).map(f).reduce(add)
    print("Pi is roughly %f" % (4.0 * count / n))

    spark.stop()