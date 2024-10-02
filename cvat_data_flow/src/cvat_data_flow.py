"""
The cvat_data_flow module provides the CVATDataFlow class 
for downloading and building datasets from CVAT.
"""

import os
import logging
import coloredlogs
from .cvat_api import CVAT_API
from .dataset_builder import CustomDataset


class CVATDataFlow:
    """
    Class for downloading and building datasets from CVAT.

    Example:
    ```
    cvat_data_flow = CVATDataFlow(
        url='http://cvat.example.com',
        login='username',
        password='password',
        raw_data_path='/path/to/save/dataset',
        projects_ids=[1, 2, 3],
        tasks_ids=[4, 5, 6],
        dataset_format='coco',
        split=[('train', 0.7), ('val', 0.2), ('test', 0.1)],
        labels_mapping={'person': 'person', 'car': 'vehicle'},
        debug=True
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
        :param raw_data_path: The path to save the downloaded data.
        :param projects_ids: The list of project IDs to download data from.
        :param tasks_ids: The list of task IDs to download data from.
        :param dataset_format: The format of the dataset to be built.
        :param split: The split ratio for train, validation, and test datasets.
        :param labels_mapping: The mapping of labels from CVAT to the desired format.
        :param debug: Flag indicating whether to enable debug mode.
        :param labels_id_mapping: Optional mapping of label IDs from CVAT to the desired format: {name: id}.
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

        self.logger = self._setup_logging()
        self.cvat_uploader = None

    def _setup_logging(self) -> logging.Logger:
        """
        Setup logging with coloredlogs.

        :return: Configured logger instance.
        """
        level = logging.DEBUG if self.debug else logging.INFO
        logging.basicConfig(level=level)
        coloredlogs.install(level=level)
        return logging.getLogger(__name__)

    def _initialize_cvat_uploader(self):
        """Initialize the CVAT API uploader."""
        self.cvat_uploader = CVAT_API(
            url=self.url, login=self.login, password=self.password, save_path=self.raw_data_path
        )

    def download_data(self, include_images: bool = False) -> None:
        """
        Download data from CVAT.

        Downloads either specific tasks or projects based on the provided IDs.
        """
        self._initialize_cvat_uploader()

        if not self.projects_ids:
            self.logger.info(f'Start downloading tasks {self.tasks_ids} ...')
            self.cvat_uploader.upload_tasks_from_cvat(tasks=self.tasks_ids, include_images=include_images)
        else:
            self.logger.info(f'Start downloading projects {self.projects_ids} ...')
            self.cvat_uploader.upload_projects_from_cvat(project_ids=self.projects_ids, include_images=include_images)

    def build_dataset(self, save_path: str = None) -> str:
        """
        Build dataset from the downloaded data.

        :param save_path: The path to save the built dataset.
        :return: The absolute path to the built dataset.
        """
        save_path = save_path or self.raw_data_path

        if not os.path.exists(self.raw_data_path):
            self.logger.error(f'Path "{self.raw_data_path}" does not exist.')
            raise FileNotFoundError(f'The specified raw data path "{self.raw_data_path}" does not exist.')

        self.logger.info(f'Building dataset in {self.format} format from {self.raw_data_path} ...')
        dataset = CustomDataset(
            datasets_path=self.raw_data_path,
            export_format=self.format,
            splits=self.split,
            mapping=self.labels_mapping,
            labels_id_mapping=self.labels_id_mapping
        )
        exported_path = dataset.export_dataset(save_path=save_path)
        self.logger.info(f'Dataset in {self.format} format has been saved to {exported_path}')

        return os.path.abspath(f'{exported_path}')