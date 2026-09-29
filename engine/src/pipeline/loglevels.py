import logging

loglevels = {
    'NOTSET': logging.NOTSET,
    'DEBUG': logging.DEBUG,
    # per-stage timings, only emitted under --benchmark; below PROGRESS so the console skips them
    'BENCHMARK': logging.INFO-8,
    'PROGRESS': logging.INFO-5,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}
