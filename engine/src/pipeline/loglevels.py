import logging

loglevels = {
    'NOTSET': logging.NOTSET,
    'DEBUG': logging.DEBUG,
    'PROGRESS': logging.INFO-5,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}
