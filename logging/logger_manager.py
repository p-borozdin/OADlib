""" The module provides the architecture of a LoggerManager class
"""
import logging
import io


class LoggerManager:
    """ Class representing an entity to manage logging whatever you want
    into a log file
    """

    __levels = {
        logging.CRITICAL,
        logging.FATAL,
        logging.ERROR,
        logging.WARNING,
        logging.WARN,
        logging.INFO,
        logging.DEBUG,
        logging.NOTSET,
        }

    def __init__(
            self,
            level: int = logging.INFO,
            name: str = "logger"
            ):
        """ The logger manager initialization

        Args:
            level (int, optional): logger's level. Defaults to `logging.INFO`.
            name (str, optional): logger's name. Defaults to `"logger"`.
        """
        assert level in self.__levels, \
            f"Unable to create logger: level must be in {self.__levels}, " \
            f"but got level = '{level}'"

        self.level = level
        self.name = name
        self.logger: logging.Logger | None = None
        self.ofstream: io.TextIOWrapper | None = None

    def get_name(self) -> str:
        """ Get the logger's name

        Returns:
            str: logger's name
        """
        return self.name

    def get_level(self) -> int:
        """ Get the logger's level

        Returns:
            int: logger's level
        """
        return self.level

    def bind(self, file: io.TextIOWrapper):
        """ Bind logger to a file which will be used to log to

        Args:
            file (io.TextIOWrapper): log file
        """
        assert not file.closed, \
            f"Unable to bind logger '{self.get_name()}' to a closed file"
        assert self.ofstream is None, \
            f"Unable to bind logger '{self.get_name()}' to file: " \
            f"it is already bound to a another file"

        self.ofstream = file

        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(self.level)
        handler = logging.StreamHandler(self.ofstream)
        self.logger.addHandler(handler)

    def get_logger(self) -> logging.Logger:
        """ Get the logger

        Returns:
            logging.Logger: logger
        """
        assert self.logger is not None and self.ofstream is not None, \
            f"Unable to get logger '{self.get_name()}': " \
            f"is was not bound to any file"

        return self.logger

    def close_file(self):
        """ Closes log file. Expected to be manually called
        at the end of a process
        """
        if self.ofstream is not None:
            self.ofstream.close()
