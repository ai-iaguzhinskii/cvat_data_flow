import os
import shutil
import zipfile
import json
import logging
from typing import List, Union
from logging import FileHandler

from cvat_sdk import make_client
from cvat_sdk.core.proxies.tasks import Task
from cvat_sdk.core.proxies.projects import Project
from cvat_sdk.api_client.models import PatchedTaskWriteRequest
from tqdm import tqdm


class CVAT_API:
    """
    A class used to:
    1. Upload datasets from CVAT in Datumaro format by project or task ids.
    2. Backup projects and tasks from CVAT.
    3. Restore tasks to CVAT from backup.
    4. Move tasks to a different projects.


    Example:
    ```
    cvat_uploader = CVATUploader(
        url='http://cvat.example.com',
        login='username',
        password='password',
        save_path='/path/to/save/dataset'
    )
    cvat_uploader.upload_projects_from_cvat(project_ids=[1, 2, 3])
    ```
    """

    def __init__(self, url: str, login: str, password: str, save_path: str):
        """
        :param url: CVAT url.
        :param login: CVAT login.
        :param password: CVAT password.
        :param save_path: Path to the directory where the dataset will be saved.
        """
        self.logger = self._setup_logger()
        self.client = self._connect_to_cvat(url, login, password)
        self.downloads_directory = self._setup_save_directory(save_path)
        self.img_num = 0

    @staticmethod
    def _setup_logger() -> logging.Logger:
        """Setup logger with file handler."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        file_handler = FileHandler('cvat_uploader.log')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        return logger

    def _connect_to_cvat(self, url: str, login: str, password: str):
        """Establish a connection to CVAT."""
        self.logger.info(f'Connecting to CVAT "{url}" ...')
        client = make_client(host=url, credentials=(login, password))
        return client

    @staticmethod
    def _setup_save_directory(save_path: str) -> str:
        """Create the directory for saving datasets if it doesn't exist."""
        os.makedirs(save_path, exist_ok=True)
        return save_path

    def _process_data(self, path: str) -> None:
        """Rename images and update annotation file."""
        annotation_path = os.path.join(path, 'annotations', 'default.json')
        images_path = os.path.join(path, 'images')

        with open(annotation_path, 'r') as f:
            data = json.load(f)

        for item in data['items']:
            old_image_path = os.path.join(images_path, item['image']['path'])
            new_image_path = os.path.join(images_path, f'{self.img_num}.jpg')
            os.rename(old_image_path, new_image_path)

            item['image']['path'] = item['id'] = item['media']['path'] = f'{self.img_num}.jpg'
            self.img_num += 1

        with open(annotation_path, 'w') as f:
            json.dump(data, f, indent=4)

    def _download_and_extract(self, cvat_object: Union[Task, Project]) -> None:
        """Download and extract dataset from CVAT."""
        archive_path = os.path.join(self.downloads_directory, f'{cvat_object.id}.zip')

        self.logger.debug(f'Downloading "{archive_path}" ...')
        try:
            cvat_object.export_dataset('Datumaro 1.0', archive_path)
        except Exception as e:
            self.logger.error(f'Failed to download "{archive_path}": {e}')
            return

        self._extract_dataset(archive_path)

    def _extract_dataset(self, archive_path: str) -> None:
        """Extract the downloaded dataset archive."""
        target_directory = os.path.join(self.downloads_directory, os.path.splitext(os.path.basename(archive_path))[0])
        os.makedirs(target_directory, exist_ok=True)

        self.logger.debug(f'Extracting "{archive_path}" to "{target_directory}" ...')

        try:
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(target_directory)

            source_images_dir = os.path.join(target_directory, 'images', 'default')
            target_images_dir = os.path.join(target_directory, 'images')

            for image in os.listdir(source_images_dir):
                shutil.move(os.path.join(source_images_dir, image), target_images_dir)

            shutil.rmtree(source_images_dir)
        except Exception as e:
            self.logger.error(f'Failed to extract "{archive_path}": {e}')
            os.remove(archive_path)
            return

        self.logger.debug(f'Successfully extracted "{archive_path}"')
        os.remove(archive_path)

    def _backup(self, cvat_object: Union[Task, Project]) -> None:
        """Backup project or task from CVAT."""
        archive_path = os.path.join(self.downloads_directory, f'{cvat_object.id}.zip')

        self.logger.debug(f'Backing up "{archive_path}" ...')
        try:
            cvat_object.download_backup(archive_path)
        except Exception as e:
            self.logger.error(f'Failed to backup "{archive_path}": {e}')
            return

        self.logger.debug(f'Successfully backed up "{archive_path}"')

    def _get_project_tasks(self, project_id: int) -> List[Task]:
        """Retrieve tasks from the given project."""
        project = self.client.projects.retrieve(int(project_id))
        tasks = project.get_tasks()
        self.logger.info(f'Found {len(tasks)} tasks in project "{project_id}"')
        return tasks

    def upload_tasks_from_cvat(self, tasks: List[Union[Task, int]]) -> None:
        """Upload tasks from CVAT."""
        tasks = [self.client.tasks.retrieve(int(task)) if not isinstance(task, Task) else task for task in tasks]

        for task in tqdm(tasks):
            self._download_and_extract(task)

    def upload_projects_from_cvat(self, project_ids: List[int]) -> None:
        """Upload projects from CVAT."""
        for project_id in tqdm(project_ids):
            tasks = self._get_project_tasks(project_id)
            self.upload_tasks_from_cvat(tasks)

        self.logger.info(f'Finished uploading projects {project_ids} from CVAT.')

    def backup_tasks_from_cvat(self, tasks: List[Union[Task, int]]) -> None:
        """Backup tasks from CVAT."""
        tasks = [self.client.tasks.retrieve(task) if not isinstance(task, Task) else task for task in tasks]

        for task in tqdm(tasks):
            self._backup(task)

    def backup_projects_from_cvat(self, project_ids: List[int]) -> None:
        """Backup projects from CVAT."""
        main_directory = self.downloads_directory
        try:
            for project_id in tqdm(project_ids):
                project_directory = os.path.join(main_directory, str(project_id))
                os.makedirs(project_directory, exist_ok=True)
                self.downloads_directory = project_directory
                tasks = self._get_project_tasks(project_id)
                self.backup_tasks_from_cvat(tasks)

            self.logger.info(f'Finished backing up projects {project_ids} from CVAT.')
        except Exception as e:
            self.logger.error(f'Error during backup: {e}')
            raise
        finally:
            self.downloads_directory = main_directory

    def create_task_from_backup(self, archive_path: str, organization_slug: str = 'RegularXRAY', project_id: int = 2) -> None:
        """Restore tasks to CVAT from backup."""
        self.client.organization_slug = organization_slug
        task = self.client.tasks.create_from_backup(archive_path)

        if project_id:
            patched_task = PatchedTaskWriteRequest(project_id=project_id)
            task.update(patched_task)

    def create_tasks_from_backups(self, backups_path: str, organization_slug: str = 'RegularXRAY', project_id: int = None) -> None:
        """Restore tasks to CVAT from backups."""
        backups = os.listdir(backups_path)
        self.logger.info(f'Found {len(backups)} backups in "{backups_path}"')

        for backup in tqdm(backups):
            archive_path = os.path.join(backups_path, backup)
            self.create_task_from_backup(archive_path, organization_slug, project_id)

        self.logger.info(f'Finished creating tasks from backups in "{backups_path}"')

    def move_tasks_to_project(self, task_ids: List[int], project_id: int) -> None:
        """Move tasks to a different project."""
        for task_id in tqdm(task_ids):
            task = self.client.tasks.retrieve(task_id)
            patched_task = PatchedTaskWriteRequest(project_id=project_id)
            task.update(patched_task)

        self.logger.info(f'Moved tasks {task_ids} to project "{project_id}"')