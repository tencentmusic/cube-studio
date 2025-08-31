from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession.builder \
    .appName('spark-hive-demo') \
    .config("hive.metastore.uris", "thrift://xxx.xxx.xxx.xxx:9083") \
    .enableHiveSupport() \
    .getOrCreate()

    spark.sql("create table if not exists demo(id bigint,name String)")

    spark.sql("insert overwrite demo values (1,'hamawhite'),(2,'song.bs')")
    spark.sql("select * from demo").show()