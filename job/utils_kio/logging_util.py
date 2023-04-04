import os
import sys
import logging


# logging util: write msg to file and stdout stream
class LoggerUtilKIO:
    def __init__(self):
        self.logger = None

    def init(self, fpath_logfile, logger_name, mode="a", need_file=True):
        # init logger; add handles
        self.logger = logging.Logger(name=logger_name, level=logging.DEBUG)
        self.add_stdout_stream_handle()
        if not need_file:
            print("don't need file log, skip")
        else:
            self.add_file_stream_handle(fpath_logfile, mode)

        print("should log")

    def add_stdout_stream_handle(self):
        # stdout stream handler
        self.streamHandler = logging.StreamHandler(sys.stdout)
        self.logger.addHandler(self.streamHandler)

    def add_file_stream_handle(self, fpath_logfile, mode):
        # add file handler
        if not fpath_logfile or len(fpath_logfile) == 0:
            print("fpath_log_file is none or empty, won't write msg to file")
            return

        log_dir = os.path.dirname(fpath_logfile)
        if len(log_dir) and not os.path.exists(log_dir):
            print("do not exist log dir: {}, will create it".format(log_dir))
            os.makedirs(log_dir)

        self.logFileHandler = logging.FileHandler(filename=fpath_logfile, mode=mode, encoding="utf8")
        fmt = logging.Formatter(fmt="%(asctime)s - %(name)s - %(levelname)s:  %(message)s",
                                datefmt='%Y-%m-%d %H:%M:%S')
        self.logFileHandler.setFormatter(fmt)
        self.logger.addHandler(self.logFileHandler)

    def info(self, msg):
        self.logger.info(msg)

    def debug(self, msg):
        self.logger.debug(msg)

    def error(self, msg):
        self.logger.error(msg)

    def warn(self, msg):
        self.logger.warn(msg)


if __name__ == "__main__":
    logger = LoggerUtilKIO()
    logger.init("/data/repos/job-template/job/xgb_simple_train_new/testing/log/train_log.log", "train_log")
    logger.info("wtf")
