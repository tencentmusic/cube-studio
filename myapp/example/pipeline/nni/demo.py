import os
import argparse
import random,time,datetime
import nni
import logging
from nni.utils import merge_parameter
import pysnooper
time.sleep(10)

@pysnooper.snoop()
def main(args):
    test_acc=random.randint(30,50)
    for epoch in range(1, 11):
        test_acc_epoch= random.randint(3,5)
        time.sleep(3)
        # if os.path.exists('train')
        test_acc+=test_acc_epoch
        # 上报当前迭代目标值
        nni.report_intermediate_result(test_acc)
    # 上报最总目标值
    nni.report_final_result(test_acc)


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


