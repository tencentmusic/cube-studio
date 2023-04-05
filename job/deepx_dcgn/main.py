# coding=utf-8
import argparse
import json
import os

from job.pkgs.utils import recur_expand_param

def main(job, pack_path, upstream_output_file, export_path):
    trainer = os.path.join(pack_path, 'dcgn_trainer') 
    freq_file = os.path.join(pack_path, "his_listen_song.freq")
    dict_file = os.path.join(pack_path, "his_listen_song.dict") 
    embed_out = os.path.join(export_path, "dcgn_embedding")
    out_model = os.path.join(export_path, "dcgn_model")

    print('pack_path:{}'.format(pack_path))
    print('export_path:{}'.format(export_path))
    print('out_model:{}'.format(out_model))
    
    job_detail = job.get('job_detail')
    if not job_detail:
        print ('job_detail not set')
        return

    train_data_args = job_detail.get('train_data_args')
    if not train_data_args:
        print ('train_data_args not set')
        return 
    
    input_file = train_data_args.get('train_dir')
    if input_file and input_file.strip():
        input_file = os.path.join(pack_path, input_file.strip())
        print("input_file is {}".format(input_file))
    else:
        print("input_file not set\n")
        return

    freq_uniform = train_data_args.get('freq_uniform')
    if freq_uniform and freq_uniform.strip():
        freq_uniform = os.path.join(pack_path, freq_uniform.strip())
        print("freq_uniform is {}".format(freq_uniform))
    else:
        print("freq_uniform not set\n")
        return    

    group_conf = train_data_args.get('group_conf')
    if group_conf and group_conf.strip():
        group_conf = os.path.join(pack_path, group_conf.strip())
        print("group_conf is {}", group_conf)
    else:
        print("group_conf not set\n")
        return    

    train_args = job.get('train_args')
    if not train_args:
        print('There is no train_args set\n')
        return

    params = train_args.get('params')
    if not params:
        print('There is no params set\n')
        return

    epoch = params.get('epoch')
    if not epoch:
        print('There is no epoch set\n')
        return 
    else:
        print('epoch: {}'.format(epoch))

    batch = params.get('batch')
    if not batch:
        print('There is no batch set\n')
        return
    else:
        print('batch: {}'.format(batch))
    
    thread = params.get('thread')
    if not thread:
        print('There is no thread set\n')
        return
    else:
        print('thread: {}'.format(thread))

    negative_sample_num = params.get('negative_sample_num')
    if not negative_sample_num:
        print('There is no negative_sample_num set\n')
        return
    else:
        print('negative_sample_num: {}'.format(negative_sample_num))

    ada_grad_alpha = params.get('ada_grad_alpha')
    if not ada_grad_alpha:
        print('There is no ada_grad_alpha set\n')
        return
    else:
        print('ada_grad_alpha: {}'.format(ada_grad_alpha))

    ada_grad_beta = params.get('ada_grad_beta')
    if not ada_grad_beta:
        print('There is no ada_grad_beta set\n')
        return
    else:
        print('ada_grad_beta: {}'.format(ada_grad_beta))

    leaky_relu_alpha = params.get('leaky_relu_alpha')
    if not leaky_relu_alpha:
        print('There is no leaky_relu_alpha set\n')
        return
    else:
        print('leaky_relu_alpha: {}'.format(leaky_relu_alpha))

    item_group_id = params.get('item_group_id')
    if not item_group_id:
        print('There is no item_group_id set\n')
        return
    else:
        print('item_group_id: {}'.format(item_group_id))

    label_group_id = params.get('label_group_id')
    if not item_group_id:
        print('There is no label_group_id set\n')
        return
    else:
        print('label_group_id: {}'.format(label_group_id))

    dim = params.get('dim')
    if not dim:
        print('There is no dim set\n')
        return
    else:
        print('dim: {}'.format(dim))

    os.system("mkdir -p {}".format(out_model))
    os.system("python {}/gen_music_dict.py {}".format(pack_path, freq_uniform))

    train_cmd = "{} \
        --in={} \
        --epoch={} \
        --batch={} \
        --thread={} \
        --config={} \
        --freq_file={} \
        --dict_file={} \
        --out_model={} \
        --negative_sample_num={} \
        --ada_grad_alpha={} --ada_grad_beta={} \
        --leaky_relu_alpha={} \
        --item_group_id={} \
        --label_group_id={} \
        --item_embedding_file={} \
        --dim='{}'".format(trainer, input_file, epoch, batch, thread,
        group_conf, freq_file, dict_file, out_model, negative_sample_num, 
        ada_grad_alpha, ada_grad_beta, leaky_relu_alpha, item_group_id,label_group_id,
        embed_out, dim)

    print(train_cmd)
    os.system(train_cmd)

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser("XGBoost model runner train component")
    arg_parser.add_argument('--job', type=str, required=True, help="模型训练任务描述json")
    arg_parser.add_argument('--pack-path', type=str, required=True, help="用户包（包含所有用户文件的目录）的挂载到容器中的路径")
    arg_parser.add_argument('--upstream-output-file', type=str, help="上游输出文件（包含路径）")
    arg_parser.add_argument('--export-path', type=str, required=True, help="数据导出目录")

    args = arg_parser.parse_args()  
    print("{} args: {}".format(__file__, args))

    job_spec = json.loads(args.job)
    print("job str: {}\n".format(args.job))
    job_spec = recur_expand_param(job_spec, args.export_path, args.pack_path)
    print("expanded job spec: {}\n".format(job_spec))
    main(job_spec, args.pack_path, args.upstream_output_file, args.export_path)
