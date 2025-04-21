import os
import argparse
import random,time,datetime
import nni
import logging
from nni.utils import merge_parameter
import pysnooper
time.sleep(10)

def get_acc(epoch):
    # 填写你的训练代码，生成准确率的值，下面的只是示例
    acc=random.randint(10*epoch,10*(epoch+1))
    return acc

# 平台只关注上报每次epoch的目标值(准确率)和最终的目标值(准确率)，关于如何得到的这两个值，用户自己把控
@pysnooper.snoop()
def main(args):
    total_acc = 0
    for epoch in range(1, 11):
        test_acc_epoch= get_acc(epoch)
        time.sleep(3)

        # 上报当前迭代目标值
        nni.report_intermediate_result(test_acc_epoch)
        # 计算最终准确率，这里试试个示例
        total_acc = test_acc_epoch

    # 上报最总目标值
    nni.report_final_result(total_acc)


def get_params():
    # 必须接收超参数为输入参数
    parser = argparse.ArgumentParser(description='nni Example')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--hidden_size', type=int, default=128, help='hidden size for training (default: 128)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for training (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.0, help='momentum for training (default: 0)')

    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    try:
        # get parameters form tuner
        tuner_params = nni.get_next_parameter()
        params = vars(merge_parameter(get_params(), tuner_params))
        print(tuner_params, params)
        main(params)
    except Exception as exception:
        print(exception)
        raise


