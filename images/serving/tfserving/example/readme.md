
启动训练，保存模型：
python mnist_saved_model.py --training_iteration 10 --model_version 1 export_dir

检查模型：
saved_model_cli show --dir export_dir/1 --all

启动服务：
tensorflow_model_server --rest_api_port=8501 --model_name=fashion_model --model_base_path=export_dir

