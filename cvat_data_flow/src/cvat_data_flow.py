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
        self, url: str, login: str, password: str, save_path: str,
        projects_ids: list, tasks_ids: list, only_build_dataset: bool,
        dataset_format: str, split: list, labels_mapping: dict, debug: bool = False,
        labels_id_mapping: dict = None
    ):
        """
        Initialize the CVATDataFlow object.

        :param url: The URL of the CVAT server.
        :param login: The login username for authentication.
        :param password: The login password for authentication.
        :param save_path: The path to save the uploaded data.
        :param projects_ids: The list of project IDs to upload data from.
        :param tasks_ids: The list of task IDs to upload data from.
        :param only_build_dataset: Flag indicating whether to only build the dataset without uploading.
        :param format: The format of the dataset to be built.
        :param split: The split ratio for train and test datasets.
        :param labels_mapping: The mapping of labels from CVAT to the desired format.
        :param debug: Flag indicating whether to enable debug mode.
        :param labels_id_mapping: The mapping of labels IDs from CVAT to the desired format: {name: id}.Example: {'person': 1, 'car': 2}
        """
        self.url = url
        self.login = login
        self.password = password
        self.save_path = save_path
        self.projects_ids = projects_ids
        self.tasks_ids = tasks_ids
        self.only_build_dataset = only_build_dataset
        self.format = dataset_format
        self.split = split
        self.labels_mapping = labels_mapping
        self.debug = debug
        self.labels_id_mapping = labels_id_mapping

        self.setup_logging()
        self.cvat_uploader = CVATUploader(
            url=self.url, login=self.login, password=self.password, save_path=self.save_path
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
        if not self.only_build_dataset:
            if len(self.projects_ids) == 0:
                self.logger.info(f'Start downloading tasks {self.tasks_ids} ...')
                self.cvat_uploader.upload_tasks_from_cvat(task_ids=self.tasks_ids)
            else:
                self.logger.info(f'Start downloading projects {self.projects_ids} ...')
                self.cvat_uploader.upload_projects_from_cvat(project_ids=self.projects_ids)

    def build_dataset(self):
        """
        Build dataset from the downloaded data.
        """
        if os.path.exists(self.save_path):
            self.logger.info('Start building dataset ...')
            dataset = CustomDataset(
                datasets_path=self.save_path, export_format=self.format,
                splits=self.split, mapping=self.labels_mapping,
                labels_id_mapping=self.labels_id_mapping
            )
            dataset.export_dataset()
        else:
            self.logger.error(f'Path "{self.save_path}" does not exist.')
        