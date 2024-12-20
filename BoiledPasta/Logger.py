# https://stackoverflow.com/questions/6760685/what-is-the-best-way-of-implementing-singleton-in-python#6798042
from Utilities import Singleton
import logging
from enum import Enum
from typing import LiteralString
import traceback

logging_map = {
    'NOTSET' : logging.NOTSET,
    'DEBUG' : logging.DEBUG,
    'INFO' : logging.INFO,
    'WARNING' : logging.WARNING,
    'ERROR' : logging.ERROR,
    'CRITICAL' : logging.CRITICAL
    }

logging_filepath = "./Logging/log.log"
logger = logging.getLogger(__name__)
# https://stackoverflow.com/questions/11581794/how-do-i-change-the-format-of-a-python-log-message-on-a-per-logger-basis
logger.setLevel(logging.DEBUG)
# create file handler that logs debug and higher level messages
spam_filepath = './Logging/spam.log'
fh = logging.FileHandler(spam_filepath)
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)

# add the handlers to logger
logger.addHandler(ch)
logger.addHandler(fh)

#clear the files before logging
with open(logging_filepath, 'w') as f:
    pass
    
with open(spam_filepath, 'w') as f:
    pass

class Logger(metaclass=Singleton):
    
    def __init__(self, logging_level:str):
        FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(format=FORMAT, filename=logging_filepath, level=logging_map[logging_level])
        
    def module_from_traceback(self):
        tb_stack = traceback.format_stack(limit = 3)
        filepath = tb_stack[0].split(',')[0].split()[1].strip('"')
        module_name = filepath.split('\\')[-1].split('.')[0]
        return module_name
        # return tb_stack
    
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
