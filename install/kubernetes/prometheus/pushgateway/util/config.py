
import logging
import logging.handlers
import os

# 错误码
ERROR_FILE_LARGE=1


LOCAL_SERVER_IP = "0.0.0.0"       # 服务器的ip地址
LOCAL_SERVER_PORT = 80   # 服务器的端口号
LOCAL_SERVER_KEY = '123456'    # 华为发送来的常数秘钥
LOCAL_SERVER_SIZE = 5    # 图片大小限制  5M

# response响应字典，根据arctern_req和result和response生成arctern_reply
def write_response(error, message, result):
    response = {
        'error': error,  # 错误编号，成功标号为0
        'message': message,  # 错误或成功描述。字符串
        'result': result  # 成功的返回结果,字典格式
    }
    return response

def init_logger(parent_path='/intellif/log/', sub_path='',module_name=''):
    print('init_logger')
    """
    usage: output to logfile and console simultaneous
    """
    path = parent_path + sub_path
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except Exception as e:
            print('please excute with root privilege, makdir error %s' % e)

    # create a logger
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.INFO)   # 最低输出等级

    # create log file name
    if module_name == '':
        log_file = path + 'server.log'
    else:
        log_file = path + module_name + '.log'

    try:
        # define log format
        # formatter = logging.Formatter(
        #     '%(asctime)s %(filename)s[line:%(lineno)d] ' +
        #     '%(levelname)s  %(message)s', '%Y-%m-%d %H:%M:%S')
        formatter = logging.Formatter(
            '%(asctime)s %(levelname)s  %(message)s',
            '%H:%M:%S')

        # 定义一个1小时换一次log文件的handler
        # filename是输出日志文件名的前缀
        # backupCount是保留日志个数。默认的0是不会自动删除掉日志。若设10，则在文件的创建过程中库会判断是否有超过这个10，若超过，则会从最先创建的开始删除
        # interval是指等待多少个单位when的时间后，Logger会自动重建文件，当然，这个文件的创建取决于filename + suffix，若这个文件跟之前的文件有重名，则会自动覆盖掉以前的文件，所以有些情况suffix要定义的不能因为when而重复。
        hdlr = logging.handlers.TimedRotatingFileHandler(filename=log_file, when='H', interval=1, backupCount=48)
        hdlr.setLevel(logging.INFO)    # 当前log输出等级
        hdlr.setFormatter(formatter)
        # 设置后缀名称，跟strftime的格式一样
        hdlr.suffix = "%Y-%m-%d_%H-%M-%S.log"
        logger.addHandler(hdlr)

        # create a streamhandler to output to console
        # ch = logging.StreamHandler()
        # ch.setLevel(logging.INFO)  # 当前log输出等级
        # ch.setFormatter(formatter)
        # logger.addHandler(ch)

        # create a filehandler to record log to file
        # fh = logging.FileHandler(log_file)
        # fh.setLevel(logging.INFO)
        # fh.setFormatter(formatter)
        # logger.addHandler(fh)

        logger.info(logger.name)
    except Exception as e:
        print(e)
        logging.info(
            'please execute with root privilege, init logger error %s' % e)


def init_console_logger():
    """
    usage: only output to console
    """
    # create a logger
    print('init_console_logger')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)   # 最低输出等级

    # create a streamhandler to output to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)   # 当前输出流输出等级

    # define log format
    # formatter = logging.Formatter(
    #     '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s  %(message)s',
    #     '%Y-%m-%d %H:%M:%S')
    formatter = logging.Formatter(
        '%(asctime)s %(levelname)s  %(message)s',
        '%H:%M:%S')
    ch.setFormatter(formatter)

    # add handler
    logger.addHandler(ch)
