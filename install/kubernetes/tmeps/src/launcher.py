# -*- coding: utf-8 -*-
import common
import numpy as np
import tensorflow as tf
import tensorflow_recommenders_addons as tfra
import json
import time
import os,sys
import traceback
import logging

from input_fn_builder import *
from model_fn_builder import *

def redis_config_init():
	redis_config_file_path = os.getenv('model_tfra_redis_config_path')
	if not redis_config_file_path:
		raise RuntimeError("redis_config_file_path not found")
	logging.info(redis_config_file_path)

	with open(redis_config_file_path, 'r') as f:
		redis_config_context = f.read()
		if not redis_config_context:
			raise RuntimeError("redis_config_context not found")
		logging.info(redis_config_context)
	return redis_config_context

def tf_config_init():
	tf_config = json.loads(os.environ.get('ps_cluster_config') or '{}')
	if not tf_config:
		raise RuntimeError("tf_config not found")

	#"task": {"type": "evaluator", "index": 0}
	hostname = os.getenv('HOSTNAME')
	node_type = hostname.split('-')[-2]
	index = hostname.split('-')[-1]
	if (node_type not in ('chief','ps','evaluator','worker','saver')):
		raise RuntimeError("hostname illegal")
	if not(index) or int(index) < 0:
		raise RuntimeError("hostname illegal")
	tf_config["task"] = {"type": node_type, "index": int(index)}
	logging.info(tf_config)
	os.environ['TF_CONFIG'] = json.dumps(tf_config)
	return tf_config

def model_path_init():	
	model_path = os.getenv('model_path') + '/'
	if (not model_path) or (not os.path.exists(model_path)):
		raise RuntimeError("model_path not found")
	logging.info(model_path)
	return model_path

def checkpoint_path_init():  
    checkpoint_path = os.getenv('checkpoint_path') + '/' 
    if (not checkpoint_path) or (not os.path.exists(checkpoint_path)):
        raise RuntimeError("checkpoint_path not found")
    logging.info(checkpoint_path)
    return checkpoint_path

def data_path_init():	
	data_path = os.getenv('data_path') + '/'
	if (not data_path) or (not os.path.exists(data_path)):
		raise RuntimeError("data_path not found")
	logging.info(data_path)
	return data_path

def train_and_evaluate(model_dir, model_fn, input_fn, ps_num):
    logging.info("Enter train_and_evaluate...")
    cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
    cluster_spec = cluster_resolver.cluster_spec()

    if cluster_resolver.task_type in ("chief", "worker", "evaluator"):
        ps_strategy = tf.compat.v1.distribute.experimental.ParameterServerStrategy(cluster_resolver)

        model_config = tf.estimator.RunConfig(log_step_count_steps=1,
                                          save_summary_steps=100,
                                          save_checkpoints_steps=100,
                                          save_checkpoints_secs=None,
                                          keep_checkpoint_max=1,
                                          train_distribute=ps_strategy)

        estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       model_dir=model_dir,
                                       params={"ps_num": ps_num},
                                       config=model_config)

        train_spec = tf.estimator.TrainSpec(input_fn=input_fn, 
                                            max_steps=None)
        eval_spec = tf.estimator.EvalSpec(input_fn=input_fn, 
                                          start_delay_secs=600,
                                         steps=10)
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    elif cluster_resolver.task_type in ("ps"):
        this_server = tf.distribute.Server(
            cluster_spec,
            job_name=cluster_resolver.task_type,
            task_index=cluster_resolver.task_id,
            protocol="grpc")
        this_server.join()
    else:
        raise RuntimeError("who are you")

def export_for_serving(checkpoint_path, model_fn, export_dir, ps_num):
    logging.info("Enter export_for_serving...")
    def serving_input_receiver_dense_fn():
        input_spec = { 
            "movie_id": tf.constant([1], tf.int64),
            "user_id": tf.constant([1], tf.int64),
            "user_rating": tf.constant([1.0], tf.float32)
        }
        return tf.estimator.export.build_raw_serving_input_receiver_fn(input_spec)
    
    model_config = tf.estimator.RunConfig(log_step_count_steps=100,
                                          save_summary_steps=100,
                                          save_checkpoints_steps=100,
                                          save_checkpoints_secs=None)
    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       model_dir=checkpoint_path,
                                       params={"ps_num": ps_num},
                                       config=model_config)

    logging.info("Waiting 60s brfore start saving...")
    time.sleep(60)
    while True:
        start = time.time()
        estimator.export_saved_model(export_dir, serving_input_receiver_dense_fn())
        end = time.clock()
        logging.info("Saving process cost {}s".format(str(end-start)))
        logging.info("Waiting 600s brfore next saving...")
        time.sleep(600)

if __name__ == '__main__':
	model_path = model_path_init()
	checkpoint_path = checkpoint_path_init()
	data_path = data_path_init()

	tf_config = tf_config_init()
	redis_config_init()
	ps_num = len(tf_config["cluster"]["ps"])

	role = tf_config["task"]["type"]
	if role != "saver":
		try:
			train_and_evaluate(checkpoint_path, model_fn, input_fn2, ps_num)
		except Exception as e:
			traceback.print_exc()
	else:
		tfra.dynamic_embedding.enable_inference_mode()
		export_for_serving(checkpoint_path, model_fn, model_path, ps_num)

	
