# https://stackoverflow.com/questions/6760685/what-is-the-best-way-of-implementing-singleton-in-python#6798042
from Utilities import Singleton
import logging
from enum import Enum
from typing import LiteralString

formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

# levels = {"NOTSET":logging.NOTSET,
#     "DEBUG":logging.DEBUG,
#     "INFO":logging.INFO,
#     "WARNING":logging.WARNING,
#     "ERROR":logging.ERROR,
#     "CRITICAL":logging.CRITICAL}


class logging_level(Enum):
    NOTSET = 0
    DEBUG = 1
    INFO = 2
    WARNING = 3
    ERROR = 4
    CRITICAL = 5

class Logger(metaclass=Singleton):
    
    def __init__(self, module_name:str, shared_logging:bool, logging_level):
        
        # if not logging_level in levels.keys():
        #     raise ValueError("Valid log levels entered as string are NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL")
        
        if not hasattr(self, 'logging_level'):
            self.module_info = dict()
            
        # new_logger = self.setup_logger(module_name, f'Logging/{module_name}.log', level=logging_level)
        self.module_info[module_name] = {'shared_logging':shared_logging, 'logging_level':logging_level, 'filepath':f'Logging/{module_name}.log'}
                                        #  'logger':new_logger}
        
        self.clear_starting_logs = True
            
        if hasattr(self, 'logging'):
            pass
        else:
            self.shared_log_filepath = 'Logging/shared_logging.log'
            self.logging = self.setup_logger("shared_logging", self.shared_log_filepath, level=logging_level)
        
        if self.clear_starting_logs:
            for module_name in self.module_info.keys():
                open(f'Logging/{module_name}.log', 'w').close()
            open(self.shared_log_filepath, 'w').close()
                   
    def setup_logger(self, module_name:str, filename:str, level=logging_level.INFO):
        if not self.logging_info:
            self.logging_info = {}
        self.logging_info[module_name] = {'filename':filename, 'level':level}
        
    def log_to_file(self, filepath, msg):
        with open(filepath, 'a') as f:
            f.writelines
        
    def notset(self, module_name:str, msg:str):
        if self.module_info[module_name] > logging_level.NOTSET:
            return
        if module_name in self.module_names_with_shared_logging:
            self.log_to_file(self.shared_log_filepath, module_name + "::" + msg)
        filepath = self.module_info[module_name]['filepath']
        self.log_to_file(filepath, msg) 
    
    def debug(self, module_name:str, msg:str):
        if self.module_info[module_name] > logging_level.DEBUG:
            return
        if module_name in self.module_names_with_shared_logging:
            self.log_to_file(self.shared_log_filepath, module_name + "::" + msg)
        filepath = self.module_info[module_name]['filepath']
        self.log_to_file(filepath, msg)
        
    def info(self, module_name:str, msg:str):
        if self.module_info[module_name] > logging_level.INFO:
            return
        if module_name in self.module_names_with_shared_logging:
            self.log_to_file(self.shared_log_filepath, module_name + "::" + msg)
        filepath = self.module_info[module_name]['filepath']
        self.log_to_file(filepath, msg)
        
    def warning(self, module_name:str, msg:str):
        if self.module_info[module_name] > logging_level.WARNING:
            return
        if module_name in self.module_names_with_shared_logging:
            self.log_to_file(self.shared_log_filepath, module_name + "::" + msg)
        filepath = self.module_info[module_name]['filepath']
        self.log_to_file(filepath, msg)
        
    def error(self, module_name:str, msg:str):
        if self.module_info[module_name] > logging_level.ERROR:
            return
        if module_name in self.module_names_with_shared_logging:
            self.log_to_file(self.shared_log_filepath, module_name + "::" + msg)
        filepath = self.module_info[module_name]['filepath']
        self.log_to_file(filepath, msg)
    
    def critical(self, module_name:str, msg:str):
        if self.module_info[module_name] > logging_level.CRITICAL:
            return
        if module_name in self.module_names_with_shared_logging:
            self.log_to_file(self.shared_log_filepath, module_name + "::" + msg)
        filepath = self.module_info[module_name]['filepath']
        self.log_to_file(filepath, msg)
           
    # https://stackoverflow.com/questions/11232230/logging-to-two-files-with-different-settings#11233293 
    # def setup_logger(self, module_name:str, filename:str, level=logging.INFO):
    #     """To setup as many loggers as you want"""

    #     handler = logging.FileHandler(filename)        
    #     handler.setFormatter(formatter)

    #     logger = logging.getLogger(module_name)
    #     logger.setLevel(level)
    #     logger.addHandler(handler)

    #     return logger
            
    # def notset(self, module_name:str, msg:str):
    #     if module_name in self.module_names_with_shared_logging:
    #         self.logging.notset(module_name + "::" + msg)
    #     self.loggers[module_name].notset(msg) 
    
    # def debug(self, module_name:str, msg:str):
    #     if module_name in self.module_names_with_shared_logging:
    #         self.logging.debug(module_name + "::" + msg)
    #     self.loggers[module_name].debug(msg)
        
    # def info(self, module_name:str, msg:str):
    #     if module_name in self.module_names_with_shared_logging:
    #         self.logging.info(module_name + "::" + msg)
    #     self.loggers[module_name].info(msg)
        
    # def warning(self, module_name:str, msg:str):
    #     if module_name in self.module_names_with_shared_logging:
    #         self.logging.warning(module_name + "::" + msg)
    #     self.loggers[module_name].warning(msg)
        
    # def error(self, module_name:str, msg:str):
    #     if module_name in self.module_names_with_shared_logging:
    #         self.logging.error(module_name + "::" + msg)
    #     self.loggers[module_name].error(msg)
    
    # def critical(self, module_name:str, msg:str):
    #     if module_name in self.module_names_with_shared_logging:
    #         self.logging.critical(module_name + "::" + msg)
    #     self.loggers[module_name].critical(msg)
    

