"""
This file contains the Config and Options classes to parse the config file.
"""
import ast
import configparser

class Config:
    """
    Class to parse the config file.
    """
    def __init__(self, config_file: str):
        self.config = configparser.ConfigParser()
        self.config.read(config_file)

    def get(self, section: str, option: str, value_type: type):
        """
        Get the value from the config file.

        :param section: The section of the config file.
        :param option: The option of the config file.
        :param value_type: The type of the value to be returned.
        :return: The value from the config file.
        """
        if value_type == int:
            return self.config.getint(section, option)
        elif value_type == float:
            return self.config.getfloat(section, option)
        elif value_type == bool:
            return self.config.getboolean(section, option)
        elif value_type == dict:
            # If the expected type is a dictionary, use ast.literal_eval
            return ast.literal_eval(self.config.get(section, option))
        elif value_type == list:
            # Strip the [] characters and split on commas
            return [item.strip() for item in self.config.get(section, option).strip('[]').split(',') if item.strip()]
        else:
            return self.config.get(section, option)


class Options:
    """
    Class to parse the options from the config file.
    """
    def __init__(self, config_file='config.ini'):
        config = Config(config_file)

        # cvat
        self.url = config.get('CVAT', 'URL', str)
        self.login = config.get('CVAT', 'LOGIN', str)
        self.password = config.get('CVAT', 'PASS', str)

        # download
        self.tasks_ids = config.get('DOWNLOAD', 'TASKS_IDS', list)
        self.projects_ids = config.get('DOWNLOAD', 'PROJECTS_IDS', list)

        # dataset
        self.format = config.get('DATASET', 'FORMAT', str)
        self.save_path = config.get('DATASET', 'SAVE_PATH', str)
        self.split = [
            (str(key), float(value)) for key, value in config.get('DATASET', 'SPLIT', dict).items()
        ]

        # options
        self.only_build_dataset = config.get('OPTIONS', 'ONLY_BUILD_DATASET', bool)
        self.labels_mapping = [
            (str(key), str(value)) for key, value in config.get('OPTIONS', 'LABELS_MAPPING', dict).items()
        ]
        if len(self.labels_mapping) == 0:
            self.labels_mapping = None
        self.debug = config.get('OPTIONS', 'DEBUG', bool)
