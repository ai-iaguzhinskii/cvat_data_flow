"""
The cvat_data_flow module provides the CVATDataFlow class 
for downloading and building datasets from CVAT.
"""
import os
import logging
import coloredlogs
from .cvat_api import CVATUploader
from .dataset_builder import CustomDataset

class CVATDataFlow:
    """
    Class for downloading and building datasets from CVAT.

    Main methods:
    - download_data
    - build_dataset

    Download and build datasets from CVAT in the specified format and save it to the specified directory:
    Example:
    ```
    cvat_data_flow = CVATDataFlow(
        url='http://cvat.example.com',
        login='username',
        password='password',
        save_path='/path/to/save/dataset',
        projects_ids=[1, 2, 3],
        tasks_ids=[4, 5, 6],
        only_build_dataset=False,
        format='coco',
        split=[('train', 0.7), ('val', 0.2), ('test', 0.1)],
        labels_mapping={'person': 'person', 'car': 'vehicle'}
    )

    cvat_data_flow.download_data()
    cvat_data_flow.build_dataset()
    ```
    """
    def __init__(
        self, url: str, login: str, password: str, raw_data_path: str,
        projects_ids: list, tasks_ids: list,
        dataset_format: str, split: list, labels_mapping: dict, debug: bool = False,
        labels_id_mapping: dict = None
    ):
        """
        Initialize the CVATDataFlow object.

        :param url: The URL of the CVAT server.
        :param login: The login username for authentication.
        :param password: The login password for authentication.
        :param raw_data_path: The path to save the uploaded data.
        :param projects_ids: The list of project IDs to upload data from.
        :param tasks_ids: The list of task IDs to upload data from.
        :param format: The format of the dataset to be built.
        :param split: The split ratio for train and test datasets.
        :param labels_mapping: The mapping of labels from CVAT to the desired format.
        :param debug: Flag indicating whether to enable debug mode.
        :param labels_id_mapping: The mapping of labels IDs from CVAT to the desired format: {name: id}.Example: {'person': 1, 'car': 2}
        """
        self.url = url
        self.login = login
        self.password = password
        self.raw_data_path = raw_data_path
        self.projects_ids = projects_ids
        self.tasks_ids = tasks_ids
        self.format = dataset_format
        self.split = split
        self.labels_mapping = labels_mapping
        self.debug = debug
        self.labels_id_mapping = labels_id_mapping

        self.setup_logging()
        self.cvat_uploader = CVATUploader(
            url=self.url, login=self.login, password=self.password, save_path=self.raw_data_path
        )

    def setup_logging(self):
        """
        Setup logging and coloredlogs.
        """
        level = logging.DEBUG if self.debug else logging.INFO
        logging.basicConfig(level=level)
        coloredlogs.install(level=level)
        self.logger = logging.getLogger(__name__)

    def download_data(self):
        """
        Download data from CVAT.
        """
        if len(self.projects_ids) == 0:
            self.logger.info(f'Start downloading tasks {self.tasks_ids} ...')
            self.cvat_uploader.upload_tasks_from_cvat(task_ids=self.tasks_ids)
        else:
            self.logger.info(f'Start downloading projects {self.projects_ids} ...')
            self.cvat_uploader.upload_projects_from_cvat(project_ids=self.projects_ids)

    def build_dataset(self, save_path: str = None):
        """
        Build dataset from the downloaded data.

        :param save_path: The path to save the built dataset.
        :return: The path to the built dataset.
        """
        if save_path is None:
            save_path = self.raw_data_path
        if os.path.exists(self.raw_data_path):
            self.logger.info('Start building dataset ...')
            dataset = CustomDataset(
                datasets_path=self.raw_data_path, 
                export_format=self.format,
                splits=self.split, 
                mapping=self.labels_mapping,
                labels_id_mapping=self.labels_id_mapping
            )
            dataset.export_dataset(save_path=save_path)
            self.logger.info(f'Dataset in {self.format} format has been saved to {save_path}.')

            return os.path.abspath(f'{save_path}_{self.format}')
        else:
            self.logger.error(f'Path "{self.raw_data_path}" does not exist.')
        