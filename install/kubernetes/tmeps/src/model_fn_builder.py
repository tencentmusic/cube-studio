# -*- coding: utf-8 -*-
import common
import sys,os
import numpy as np
from tensorflow.keras.layers import Dense
import tensorflow as tf
import tensorflow_recommenders_addons as tfra

def model_fn(features, labels, mode, params):
	embedding_size = 36
	movie_id = features["movie_id"]
	user_id = features["user_id"]
	rating = features["user_rating"]

	if mode == tf.estimator.ModeKeys.TRAIN:
		lookup_node_list = [
			"/job:ps/replica:0/task:{}/CPU:0".format(i)
			for i in range(params["ps_num"])]
		initializer = tf.keras.initializers.RandomNormal(-1, 1)
	else:
		lookup_node_list = ["/job:localhost/replica:0/task:0/CPU:0"] * params["ps_num"]
		initializer = tf.keras.initializers.Zeros()

	redis_config=tfra.dynamic_embedding.RedisTableConfig(
		redis_config_abs_dir_env="model_tfra_redis_config_path"
	)
	redis_creator=tfra.dynamic_embedding.RedisTableCreator(redis_config)
	user_embeddings = tfra.dynamic_embedding.get_variable(
		name="user_dynamic_embeddings",
		dim=embedding_size,
		devices=lookup_node_list,
		initializer=initializer,
		kv_creator=redis_creator)
	movie_embeddings = tfra.dynamic_embedding.get_variable(
		name="moive_dynamic_embeddings",
		dim=embedding_size,
		devices=lookup_node_list,
		initializer=initializer,
		kv_creator=redis_creator)

	user_id_val, user_id_idx = tf.unique(tf.concat(user_id, axis=0))
	user_id_weights, user_id_trainable_wrapper = tfra.dynamic_embedding.embedding_lookup(
		params=user_embeddings,
		ids=user_id_val,
		name="user-id-weights",
		return_trainable=True)
	user_id_weights = tf.gather(user_id_weights, user_id_idx)

	movie_id_val, movie_id_idx = tf.unique(tf.concat(movie_id, axis=0))
	movie_id_weights, movie_id_trainable_wrapper = tfra.dynamic_embedding.embedding_lookup(
		params=movie_embeddings,
		ids=movie_id_val,
		name="movie-id-weights",
		return_trainable=True)
	movie_id_weights = tf.gather(movie_id_weights, movie_id_idx)

	embeddings = tf.concat([user_id_weights, movie_id_weights], axis=1)
	d0 = Dense(256,
			   activation='relu',
			   kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1),
			   bias_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1))
	d1 = Dense(64,
			   activation='relu',
			   kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1),
			   bias_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1))
	d2 = Dense(1,
			   kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1),
			   bias_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1))
	dnn = d0(embeddings)
	dnn = d1(dnn)
	dnn = d2(dnn)
	out = tf.reshape(dnn, shape=[-1])
#	loss = tf.keras.losses.MeanSquaredError()(rating, out)

	per_example_loss = (out - rating)**2
	loss = tf.nn.compute_average_loss(per_example_loss)

	predictions = {"out": out}
	acc = tf.metrics.Accuracy()
	acc.update_state([0.1, 1.0], [1.0, 0.1])

	tensors_to_log = {"user_id_val": user_id_val.name}
	hook = tf.estimator.LoggingTensorHook(tensors_to_log, every_n_iter=100)

	if mode == tf.estimator.ModeKeys.EVAL:
		eval_metric_ops = {"accuracy": acc}
		return tf.estimator.EstimatorSpec(mode=mode,
										  loss=loss,
										  eval_metric_ops=eval_metric_ops)

	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)
		optimizer = tfra.dynamic_embedding.DynamicEmbeddingOptimizer(optimizer)
		train_op = optimizer.minimize(
			loss, global_step=tf.compat.v1.train.get_or_create_global_step())
		return tf.estimator.EstimatorSpec(mode=mode,
										  predictions=predictions,
										  loss=loss,
										  train_op=train_op,
											training_hooks=[hook])

	if mode == tf.estimator.ModeKeys.PREDICT:
		predictions_for_net = {"out": out}
		export_outputs = {
			"predict_export_outputs":
				tf.estimator.export.PredictOutput(outputs=predictions_for_net)
		}
		return tf.estimator.EstimatorSpec(mode,
										  predictions=predictions_for_net,
										  export_outputs=export_outputs,
											prediction_hooks=[hook])

# if __name__ == '__main__':

