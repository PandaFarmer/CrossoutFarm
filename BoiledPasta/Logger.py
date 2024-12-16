# https://stackoverflow.com/questions/6760685/what-is-the-best-way-of-implementing-singleton-in-python#6798042
from Utilities import Singleton
import logging
from enum import Enum
from typing import LiteralString
import traceback


logging_filepath = "/logging/log.log"
logger = logging.getLogger(__name__)
# https://stackoverflow.com/questions/11581794/how-do-i-change-the-format-of-a-python-log-message-on-a-per-logger-basis
logger.setLevel(logging.DEBUG)
# create file handler that logs debug and higher level messages
fh = logging.FileHandler('/logging/spam.log')
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
# create formatter and add it to the handlers
# formatter = logging.Formatter(
#     '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

ch.setFormatter(formatter)
fh.setFormatter(formatter)
# add the handlers to logger
logger.addHandler(ch)
logger.addHandler(fh)

class Logger(metaclass=Singleton):
    
    def __init__(self, logging_level=logging.NOTSET):
        logging.basicConfig(filename=logging_filepath, level=logging.INFO)
        
        
    def module_from_traceback(self):
        tb_stack = traceback.format_stack(limit = 2)
        rv = tb_stack[0].split(',')[0].split()[1].strip('"')
        assert('<' in rv and '>' in rv)
        return rv
        
    def info(self):
        module = self.module_from_traceback()
    
    def notset(self, msg):
        module = self.module_from_traceback()
        logging.notset(f"{module} :: {msg}")

    def debug(self, msg):
        module = self.module_from_traceback()
        logging.debug(f"{module} :: {msg}")

    def info(self, msg):
        module = self.module_from_traceback()
        logging.info(f"{module} :: {msg}")

    def warning(self, msg):
        module = self.module_from_traceback()
        logging.warning(f"{module} :: {msg}")

    def error(self, msg):
        module = self.module_from_traceback()
        logging.error(f"{module} :: {msg}")

    def critical(self, msg):
        module = self.module_from_traceback()
        logging.critical(f"{module} :: {msg}")
