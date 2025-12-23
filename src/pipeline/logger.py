import logging

from src.pipeline.assignments import AssignLog
from src.pipeline.coordination import Coordinator
from src.pipeline.loglevels import loglevels


class FilterDropProgress(logging.Filter):
    def filter(self, record):
        return record.levelno != 'PROGRESS'


class WorkerLogger:
    def __init__(self,
                 path_log,
                 coordinator: Coordinator,
                 verbosity_print: str='PROGRESS',
                 verbosity_log: str="DEBUG",
                 log_progress: bool=False):

        self.path_log = path_log
        self.coordinator = coordinator
        self.verbosity_print_int = loglevels[verbosity_print]

        self.log = logging.getLogger('buzzdetect')
        self.log.setLevel('DEBUG')

        self.format_file = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        self.handle_file = logging.FileHandler(path_log)
        self.handle_file.setLevel(verbosity_log)
        if not log_progress:
            self.handle_file.addFilter(FilterDropProgress())
        self.handle_file.setFormatter(self.format_file)
        self.log.addHandler(self.handle_file)

        self.handle_console = logging.StreamHandler()
        self.handle_console.setLevel(self.verbosity_print_int)
        self.log.addHandler(self.handle_console)

    def __call__(self):
        self.run()

    def write_log(self, a_log):
        self.log.log(msg=a_log.message, level=a_log.level_int)
        if self.coordinator.q_gui is not None and a_log.level_int >= self.verbosity_print_int:
            self.coordinator.q_gui.put(a_log)

    def run(self):
        a_log = self.coordinator.q_log.get()
        while not a_log.terminate:
            self.write_log(a_log)
            a_log = self.coordinator.q_log.get()

        self.write_log(AssignLog(message='logger closing', level_str='DEBUG'))


