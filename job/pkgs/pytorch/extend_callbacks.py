# coding=utf-8
# @Time     : 2020/12/31 10:45
# @Auther   : lionpeng@tencent.com

import ignite.engine as ign_engine
import time


class TrainSpeedLoggerCallback(object):
    def __init__(self, every_batches=100):
        self.every_batches = max(int(every_batches), 1)
        self.total_batches = 0
        self.total_time = 0
        self.start_time = None

    def attach(self, trainer: ign_engine.Engine):
        trainer.add_event_handler(ign_engine.Events.GET_BATCH_STARTED, self.on_train_batch_begin)
        trainer.add_event_handler(ign_engine.Events.ITERATION_COMPLETED, self.on_train_batch_end)

    def on_train_batch_begin(self, trainer: ign_engine.Engine):
        if self.start_time is None:
            self.start_time = time.perf_counter()

    def on_train_batch_end(self, trainer: ign_engine.Engine):
        batch = trainer.state.iteration
        self.total_batches += 1
        if self.total_batches % self.every_batches == 0:
            elapsed = time.perf_counter() - self.start_time
            self.total_time += elapsed
            rt_speed = self.every_batches / elapsed
            avg_speed = self.total_batches / self.total_time
            print("step {}(batch #{}): total cost time {}s, {} batchs cost {}s, rt step/sec: {}, avg step/sec: {}"
                  .format(self.total_batches, batch, self.total_time, self.every_batches, elapsed, rt_speed,
                          avg_speed))
            self.start_time = None
