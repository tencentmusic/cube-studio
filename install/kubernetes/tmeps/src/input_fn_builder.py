# -*- coding: utf-8 -*-
import common
import sys,os
import json
import time
import traceback
import csv
import random
import string
import kafka
from kafka import KafkaConsumer
from google.protobuf import json_format
from sample_pb2 import Sample

import tensorflow as tf
import tensorflow_datasets as tfds

def data_map_fn(x):
	return {
			"movie_id": tf.strings.to_number(x["movie_id"], tf.int64),
			"user_id": tf.strings.to_number(x["user_id"], tf.int64),
			"user_rating": x["user_rating"]
		}

def input_fn():
	data_path = os.getenv('data_path') + '/' 
	dataset = tfds.load("movielens/100k-ratings", split="train", data_dir=data_path).repeat(1000000)
	dataset = dataset.map(data_map_fn)
	#dataset = dataset.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
	dataset = dataset.batch(10)
	return dataset

def data_gen_init():
	data_path = os.getenv('data_path') + '/' 
	if (not data_path) or (not os.path.exists(data_path)):
		raise RuntimeError("data_path not found")
	dataset = tfds.load("movielens/100k-ratings", split="train", data_dir=data_path).repeat(1000000)
	dataset = dataset.map(data_map_fn).shuffle(buffer_size=256)
	dataset_iter = dataset.as_numpy_iterator()
	for i, line in enumerate(dataset_iter):
		time.sleep(0.1)
#		print("get one line : {}".format(line))
		yield line

def input_fn2():
	dataset = tf.data.Dataset.from_generator(generator = data_gen_init,
												 output_signature={
													'movie_id':tf.TensorSpec(shape=(), dtype=tf.int64),
													'user_id':tf.TensorSpec(shape=(), dtype=tf.int64),
													'user_rating':tf.TensorSpec(shape=(), dtype=tf.float32)}).batch(10)
	#dataset = dataset.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
	return dataset

class CdmqPollError(BaseException):
	def __init__(self):
		self.msg = 'poll没从cdmq拉取到数据'
	def __str__(self):
		return self.msg

def cdmq_gen_init():
	cdmq_configs = {
		"bootstrap_servers": "cdmqszentry01.data.mig:10005,cdmqszentry02.data.mig:10069,cdmqszentry05.data.mig:10033,cdmqszentry06.data.mig:10021",
		"client_id": "cg_VIDEOSAMPLE",
		"group_id": "cg_VIDEOSAMPLE",
		'enable_auto_commit': True,  # 是否自动提交消费进度，默认为true。
		'auto_commit_interval_ms': 3000,  # 自动提交消费进度的间隔，默认为5000毫秒。
		'auto_offset_reset': 'latest',  # 一个消费组在首次消费时，从哪个位置开始拉取消息，默认为latest（即最新的位置）。
		'max_poll_records': 100,  # 单次拉取（poll）的最大消息条数，默认500条。
		'fetch_max_bytes': 1048576,  # 单次拉取（poll）的最大消息大小，根据经验，这里设置100K~1M比较合适。
		'max_partition_fetch_bytes': 524288,  # 单次拉取请求中，单个分区最大返回消息大小。一次拉取请求可能返回多个分区的数据，这里限定单个分区的最大数据大小。
		'fetch_max_wait_ms': 100,  # 单次拉取请求最长等待时间，最长等待时间仅在没有最新数据时才会等待。此值应当设置较大点，减少空请求对服务端QPS的消耗。
	}
	cdmq_topic = "U_TOPIC_VIDEOSAMPLE"

	consumer = None
	while not consumer:
		try:
			consumer = KafkaConsumer(cdmq_topic, **cdmq_configs)
		except kafka.errors.NoBrokersAvailable as ke:
			time.sleep(1)
			logging.error("kafka.errors.NoBrokersAvailable raised, retry...")
	print(consumer)

	def handle_msgs(msgs):
		for tp, records in msgs.items():
			for msg in records:
				msg_dict = handle_msg(msg.value)
				return msg_dict

	def handle_msg(msg):
		sample = Sample()
		sample.ParseFromString(msg)
		sample_dict = json_format.MessageToDict(sample)
		csv_dict = sample_dict['csv']
		return csv_dict

	def mask(msg_dict):
		res = {}
		res['user_rating'] = msg_dict['qv_avg_pt_7d']
		res['user_id'] = msg_dict['sex']
		res['movie_id'] = msg_dict['age']
		return res

	while True:
		try:
			msgs = consumer.poll(timeout_ms=200, max_records=1)
			if not msgs:
				raise(CdmqPollError)
			msg_dict = handle_msgs(msgs)

			msg_dict = mask(msg_dict)
			yield msg_dict

		except CdmqPollError as e:
			time.sleep(3)
		except Exception as e:
			logging.error(traceback.format_exc())
		finally:
			time.sleep(0.1)

def input_fn3():
	dataset = tf.data.Dataset.from_generator(generator = cdmq_gen_init,
												 output_signature={
													'movie_id':tf.TensorSpec(shape=(), dtype=tf.int64),
													'user_id':tf.TensorSpec(shape=(), dtype=tf.int64),
													'user_rating':tf.TensorSpec(shape=(), dtype=tf.float32)}).batch(256)
#	dataset = dataset.map(data_map_fn)
	return dataset

def inc_gen():
	for i in range(10000000):
		if i % 1000 == 0:
			time.sleep(0.1)
		res = {}
		res['user_rating'] = 0.001
		res['user_id'] = i
		res['movie_id'] = 100000000 - i
		yield res

def input_fn4():
    dataset = tf.data.Dataset.from_generator(generator = inc_gen,
                                                 output_signature={
                                                    'movie_id':tf.TensorSpec(shape=(), dtype=tf.int64),
                                                    'user_id':tf.TensorSpec(shape=(), dtype=tf.int64),
                                                    'user_rating':tf.TensorSpec(shape=(), dtype=tf.float32)}).batch(10000)
    return dataset

if __name__ == '__main__':
#	dataset = input_fn()
#	tf.print(list(dataset.take(1)))
#	dataset = input_fn3()
#	tf.print(list(dataset.take(1)))
	dataset = input_fn2()
	tf.print(list(dataset.take(1)))


