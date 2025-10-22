from multiprocessing import Event, Process
from queue import Queue
from assignments import AssignLog

def early_exit(msg: str, level: str, event_analysisdone: Event, event_closelogger: Event, proc_logger: Process, q_log: Queue):
    q_log.put(AssignLog(msg=msg, level_str=level))
    event_closelogger.set()
    event_analysisdone.set()
    proc_logger.join()


def run_worker(workerclass, **kwargs):
    worker = workerclass(**kwargs)
    worker()
