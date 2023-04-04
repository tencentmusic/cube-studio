  
  
| 别名 | 框架支持 | 说明 |  
| ------ | ------ | ------ |  
| acc<br>accuracy | Tensorflow | tensorflow说明：https://www.tensorflow.org/versions/r2.3/api_docs/python/tf/keras/metrics/Accuracy |  
| auc | Tensorflow | tensorflow说明：https://www.tensorflow.org/versions/r2.3/api_docs/python/tf/keras/metrics/AUC |  
|bacc<br>bin_acc<br>binary_acc<br>binary_accuracy|Tensorflow|tensorflow说明：https://www.tensorflow.org/versions/r2.3/api_docs/python/tf/keras/metrics/BinaryAccuracy|  
|bce<br>bin_crossentropy<br>bin_cross_entropy<br>binary_crossentropy<br>binary_cross_entropy|Tensorflow|tensorflow说明：https://www.tensorflow.org/versions/r2.3/api_docs/python/tf/keras/metrics/BinaryCrossentropy|  
|cacc<br>cate_acc<br>cate_accuracy<br>categorical_acc<br>categorical_accuracy|Tensorflow|tensorflow说明：https://www.tensorflow.org/versions/r2.3/api_docs/python/tf/keras/metrics/CategoricalAccuracy|  
|cce<br>cate_crossentropy<br>cate_cross_entropy<br>categorical_crossentropy<br>categorical_cross_entropy|Tensorflow|tensorflow说明：https://www.tensorflow.org/versions/r2.3/api_docs/python/tf/keras/metrics/CategoricalCrossentropy|  
|ch<br>chinge<br>cate_hinge<br>catehinge<br>categorical_hinge|Tensorflow|tensorflow说明：https://www.tensorflow.org/versions/r2.3/api_docs/python/tf/keras/metrics/CategoricalHinge|  
|cs<br>cos_similarity<br>cossimilarity<br>cos_sim<br>cossim<br>cosine_similarity<br>|Tensorflow|tensorflow说明：https://www.tensorflow.org/versions/r2.3/api_docs/python/tf/keras/metrics/CosineSimilarity|  
|kld<br>kullback_leibler_divergence|Tensorflow|tensorflow说明：https://www.tensorflow.org/versions/r2.3/api_docs/python/tf/keras/metrics/KLDivergence|  
|mae<br>mean_abs_error<br>mean_absolute_error|Tensorflow|tensorflow说明：https://www.tensorflow.org/versions/r2.3/api_docs/python/tf/keras/metrics/MeanAbsoluteError|  
|mse<br>mean_square_error<br>mean_squared_error|Tensorflow|tensorflow说明：https://www.tensorflow.org/versions/r2.3/api_docs/python/tf/keras/metrics/MeanSquaredError|  
|mape<br>mean_abs_percent_error<br>mean_absolute_percentage_error|Tensorflow|tensorflow说明：https://www.tensorflow.org/versions/r2.3/api_docs/python/tf/keras/metrics/MeanAbsolutePercentageError|  
|msle<br>mean_square_log_error<br>mean_squared_log_error<br>mean_squared_logarithmic_error|Tensorflow|tensorflow说明：https://www.tensorflow.org/versions/r2.3/api_docs/python/tf/keras/metrics/MeanSquaredLogarithmicError|  
|prec<br>precision|Tensorflow|tensorflow说明：https://www.tensorflow.org/versions/r2.3/api_docs/python/tf/keras/metrics/Precision|  
|prec@recall<br>precision@recall<br>prec_at_recall<br>precision_at_recall|Tensorflow|tensorflow说明：https://www.tensorflow.org/versions/r2.3/api_docs/python/tf/keras/metrics/PrecisionAtRecall|  
|recall|Tensorflow|tensorflow说明：https://www.tensorflow.org/versions/r2.3/api_docs/python/tf/keras/metrics/Recall|  
|recall@prec<br>recall@precision<br>recall_at_prec<br>recall_at_precision|Tensorflow|tensorflow说明：https://www.tensorflow.org/versions/r2.3/api_docs/python/tf/keras/metrics/RecallAtPrecision|  
|rmse<br>root_mean_square_error<br>root_mean_squared_error|Tensorflow|tensorflow说明：https://www.tensorflow.org/versions/r2.3/api_docs/python/tf/keras/metrics/RootMeanSquaredError|  
|scacc<br>sparse_cate_acc<br>sparse_cate_accuracy<br>sparse_categorical_acc<br>sparse_categorical_accuracy|Tensorflow|tensorflow说明：https://www.tensorflow.org/versions/r2.3/api_docs/python/tf/keras/metrics/SparseCategoricalAccuracy|  
|scce<br>sparse_cate_crosse_entropy<br>sparse_categorical_cross_entropy<br>sparse_cate_crossentropy<br>sparse_categorical_crossentropy|Tensorflow|tensorflow说明：https://www.tensorflow.org/versions/r2.3/api_docs/python/tf/keras/metrics/SparseCategoricalCrossentropy|  
|s_top_k_cacc<br>sparse_top_k_cacc<br>sparse_top_k_cate_acc<br>sparse_top_k_cate_accuracy<br>sparse_top_k_categorical_acc<br>sparse_top_k_categorical_accuracy|Tensorflow|tensorflow说明：https://www.tensorflow.org/versions/r2.3/api_docs/python/tf/keras/metrics/SparseTopKCategoricalAccuracy|  
|top_k_cacc<br>top_k_cate_acc<br>top_k_cate_accuracy<br>top_k_categorical_acc<br>top_k_categorical_accuracy|Tensorflow|tensorflow说明：https://www.tensorflow.org/versions/r2.3/api_docs/python/tf/keras/metrics/TopKCategoricalAccuracy|  
|f1<br>f1score<br>f1_score|Tensorflow|tensorflow说明：https://www.tensorflow.org/addons/api_docs/python/tfa/metrics/F1Score|  
|gauc<br>grouped_auc|Tensorflow|按uid分组计算auc，最终值由个用户auc值加权平均得来。参数有：<br>- user_id_index：计算gauc是需要用户id的，user_id_index指定用户id在输入特征中的位置。如果输入是不命名的，则是整数序号，否则是输入名字<br>- from_logits：模型输出是否是logits（即输出没有经过sigmoid变换）- sample_size：因为gauc是需要对所有样本进行累计计算的，在训练及测试过程中，样本是逐batch读取的，为了计算效率，并不会每次读一个新batch都把现有累计的样本重新算一次，而是随机采样一部分进行计算，sample_size指定了采样大小，默认1024|  
|po_acc<br>pair_order_acc<br>pair_order_accuracy|Tensorflow|针对pairwise的训练，计算的是在给定的所有(正样本，负样本)pair中，正样本分数大于负样本的比例|  
  
