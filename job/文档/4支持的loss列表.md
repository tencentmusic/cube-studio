

| 别名 | 框架支持 | 说明 |
| ------ | ------ | ------ |
| bce<br>bin_crossentropy<br>bin_cross_entropy<br>binary_crossentropy<br>binary_cross_entropy | tensorflow<br>pytorch | tensorflow说明见：https://www.tensorflow.org/versions/r2.3/api_docs/python/tf/keras/losses/BinaryCrossentropy<br>pytorch说明见：https://pytorch.org/docs/1.4.0/nn.html#bceloss 及 https://pytorch.org/docs/1.4.0/nn.html#bcewithlogitsloss|
|cce<br>cate_crossentropy<br>cate_cross_entropy<br>categorical_crossentropy<br>categorical_crossentropy<br>categorical_cross_entropy|tensorflow<br>pytorch|tensorflow说明见：https://www.tensorflow.org/versions/r2.3/api_docs/python/tf/keras/losses/CategoricalCrossentropy<br>pytorch说明见：https://pytorch.org/docs/1.4.0/nn.html#crossentropyloss|
|ch<br>chinge<br>cate_hinge<br>catehinge<br>categorical_hinge|tensorflow<br>pytorch|tensorflow说明见：https://www.tensorflow.org/versions/r2.3/api_docs/python/tf/keras/losses/CategoricalHinge<br>pytorch说明见：https://pytorch.org/docs/1.4.0/nn.html#marginrankingloss|
|cs<br>cos_similarity<br>cossimilarity<br>cos_sim<br>cossim<br>cosine_similarity|tensorflow|tensorflow说明见：https://www.tensorflow.org/versions/r2.3/api_docs/python/tf/keras/losses/CosineSimilarity|
|hinge|tensorflow|tensorflow说明见：https://www.tensorflow.org/versions/r2.3/api_docs/python/tf/keras/losses/Hinge|
|huber|tensorflow<br>pytorch|tensorflow说明见：https://www.tensorflow.org/versions/r2.3/api_docs/python/tf/keras/losses/Huber<br>pytorch说明见：https://pytorch.org/docs/1.4.0/nn.html#smoothl1loss|
|kld<br>kullback_leibler_divergence|tensorflow<br>pytorch|tensorflow说明见：https://www.tensorflow.org/versions/r2.3/api_docs/python/tf/keras/losses/KLDivergence<br>pytorch说明见：https://pytorch.org/docs/1.4.0/nn.html#kldivloss|
|mae<br>mean_abs_error<br>mean_absolute_error|tensorflow<br>pytorch|tensorflow说明见：https://www.tensorflow.org/versions/r2.3/api_docs/python/tf/keras/losses/MeanAbsoluteError<br>pytorch说明见：https://pytorch.org/docs/1.4.0/nn.html#l1loss|
|mape<br>mean_abs_percent_error<br>mean_absolute_percentage_error|tensorflow|tensorflow说明见：https://www.tensorflow.org/versions/r2.3/api_docs/python/tf/keras/losses/MeanAbsolutePercentageError|
|mse<br>mean_square_error<br>mean_squared_error|tensorflow<br>pytorch|tensorflow说明见：https://www.tensorflow.org/versions/r2.3/api_docs/python/tf/keras/losses/MeanSquaredError<br>pytorch说明见：https://pytorch.org/docs/1.4.0/nn.html#mseloss|
|msle<br>mean_square_log_error<br>mean_squared_log_error<br>mean_squared_logarithmic_error|tensorflow|tensorflow说明见：https://www.tensorflow.org/versions/r2.3/api_docs/python/tf/keras/losses/MeanSquaredLogarithmicError|
|scce<br>sparse_cate_crosse_entropy<br>sparse_categorical_cross_entropy<br>sparse_cate_crossentropy<br>sparse_categorical_crossentropy|tensorflow|tensorflow说明见：https://www.tensorflow.org/versions/r2.3/api_docs/python/tf/keras/losses/SparseCategoricalCrossentropy|
|ctc|pytorch|pytorch说明见：https://pytorch.org/docs/1.4.0/nn.html#ctcloss|
|nll<br>neg_ll<br>neg_log_likelihood<br>negative_log_likelihood|pytorch|pytorch说明见：https://pytorch.org/docs/1.4.0/nn.html#nllloss|
|pnll<br>p_neg_ll<br>p_neg_log_likelihood<br>poisson_negative_log_likelihood|pytorch|pytorch说明见：https://pytorch.org/docs/1.4.0/nn.html#poissonnllloss|
|bpr<br>bpr_loss|tensorflow|用于pairwise训练，设s+为正样本得分，s-为负样本得分，则loss=-log(sigmoid(s+ - s-))，其中s+和s-均由模型给出|
|phinge<br>pair_hinge<br>pair_hinge_loss|tensorflow|用于pairwise训练，设s+为正样本得分，s-为负样本得分，则loss=max(0, margin - s+ + s-)，其中s+和s-均由模型给出，margin是需要设置的参数|

